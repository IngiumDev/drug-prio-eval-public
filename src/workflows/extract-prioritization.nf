#!/usr/bin/env nextflow
nextflow.enable.dsl=2


//
// Generate a NeDRex API key at runtime
//
process GenerateApiKey {
    conda "${workflow.projectDir}/../../environment.yml"
    tag 'generate_api_key'


    output:
        stdout emit: api_key

    script:
        """
        python3 ${workflow.projectDir}/../exploratory/generate_NEDREX_API_key.py \
          --base-url ${params.base_url} \
          --print-key
        """
}

//
// Download (extract) the 'drug' collection using the freshly generated key
//
process DownloadDrugList {
    conda "${workflow.projectDir}/../../environment.yml"
    tag 'download_drug_list'

    input:
      val   api_key

    output:
      path 'drug.csv', emit: drug_csv

    script:
    """
    python3 ${workflow.projectDir}/../data_downloading/nedrex_node_extraction.py \
      --base-url ${params.base_url} \
      --collections drug \
      --output ./ \
      --api-key ${api_key}
    """
}

process DownloadIndications {
    conda "${workflow.projectDir}/../../environment.yml"
    tag 'download_drug_indications'

    input:
      val api_key

    output:
      path 'drug_has_indication.csv', emit: indications_csv

    script:
    """
    python3 ${workflow.projectDir}/../data_downloading/nedrex_node_extraction.py \
      --base-url ${params.base_url} \
      --collections drug_has_indication \
      --output ./ \
      --api-key ${api_key}
    """
}

process ExtractTrueDrugs {
    conda "${workflow.projectDir}/../../environment.yml"
    tag 'extract_true_drugs'

    input:
      path indicates_csv
      val  disease_id

    output:
      path 'true_drugs.csv', emit: true_drugs_csv

    script:
    """
    python3 ${workflow.projectDir}/../exploratory/extract_true_drugs.py \
      --drug-indicates ${indicates_csv} \
      --disease-id ${disease_id} \
      --output-folder ./
    """
}

//
// Validate the downloaded list of drugs
//
process ValidateDrugs {
    conda "${workflow.projectDir}/../../environment.yml"
    tag 'validate_drugs'
    publishDir "${params.out_dir}", mode: 'copy'

    input:
      path   drug_csv
      path   candidate_file
      path   true_drugs_file

    output:
      path '*.json', emit: results_json

    script:
    """
    python3 ${workflow.projectDir}/../validation/drug_validation.py \
      --candidate   ${candidate_file} \
      --drug-list   ${drug_csv} \
      --true-drugs  ${true_drugs_file} \
      --permutation-count ${params.permutation_count} \
      ${ params.only_approved ? '--only-approved' : '' } \
      --out-dir     ./
    """
}
//
// Pipeline parameters (defaults can be overridden via CLI)
//
params.base_url           = params.base_url           ?: 'https://api.nedrex.net/licensed'
params.candidate          = params.candidate          ?: error("Missing --candidate")
params.true_drugs         = params.true_drugs         ?: null
params.out_dir            = params.out_dir            ?: params.outdir ?: 'results'
params.permutation_count  = params.permutation_count  ?: 10000
params.only_approved      = params.only_approved      ?: false
params.disease_id         = params.disease_id         ?: null


workflow {
    api_key_ch = GenerateApiKey().api_key

    download_out = DownloadDrugList(api_key_ch)

    // Determine trueDrugs channel: either provided file or generated from disease ID
    def trueDrugsCh
    if (params.true_drugs) {
        trueDrugsCh = Channel.fromPath(params.true_drugs)
    } else if (params.disease_id) {
        indications_ch = DownloadIndications(api_key_ch).indications_csv
        trueDrugsCh = ExtractTrueDrugs(indications_ch, params.disease_id).true_drugs_csv
    } else {
        error "Missing required parameter: --true_drugs or --disease_id"
    }

    ValidateDrugs(
        download_out.drug_csv,
        Channel.fromPath(params.candidate),
        trueDrugsCh
    )
}
