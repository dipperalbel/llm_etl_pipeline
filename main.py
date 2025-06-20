import json
import pandas as pd
from functools import partial
from llm_etl_pipeline import PdfConverter, Document, LocalLLM, get_filtered_fully_general_series_call_pdfs, get_series_titles_from_paths
from llm_etl_pipeline import logger
from llm_etl_pipeline import Pipeline
from llm_etl_pipeline import (
    verify_no_missing_data,
    verify_no_negatives,
    verify_no_empty_strings,
    check_numeric_columns,
    check_string_columns,
    check_columns_satisfy_regex,
    drop_rows_with_non_positive_values,
    drop_rows_not_satisfying_regex,
    drop_rows_if_no_column_matches_regex,
    remove_semantic_duplicates,
    load_df_from_json,
    verify_list_column_contains_only_ints,
    reduce_list_ints_to_unique,
    group_by_document_and_stack_types
)



# --- How to use it ---
if __name__ == "__main__":
    input_doc_path = input("Please enter the path to the directory containing the PDF documents: ")
    # CALL PAPERS DOCUMENTS HAVE MOST OF THE INFORMATION NEEDED. SO WE WILL FOCUS ON THESE. LEAVE THIS VALUE TO TRUE.
    analyze_only_call_papers=True
    ####################### IMPORTANT #######################
    # SET THE LOCAL LLM MODEL
    # FOR LLM MODEL, I USED phi4. IT GIVES GOOD AND CONSISTENT RESULTS.
    money_llm_model="phi4:14b"
    entity_llm_model="gemma3:27b"
    #########################################################
    # LOAD PDF CONVERTER OBVJECT
    pdf_converter = PdfConverter()
    #DEFINE PARAMETERS FOR THE EXTRACTION OF MONEY AND ENTITY RELATED INFORMATION. IMPORTANT STRINGS FOR MONEY ARE: (budget, grant,...). IMPORTANT STRINGS FOR ENTITY ARE ( consortium, entities, ...)
    #FOR THE REGEX, WE USE re.compile(..., re.DOTALL | re.IGNORECASE ) and re.match(...).
    reference_depth='paragraphs'
    paragraph_segmentation_mode='empty_line'
    money_regex= r"^(?=.*\d)(?=.*(?:\beur\b|\beuro\b|\beuros\b|€)).*$"
    entity_regex= r"(?=.*consortium composition)(?=.*entities)(?=.*coordinators)(?=.*beneficiaries).*"
    #DEFINE PARAMETERS FOR THE LLM. THESE PARAMETERS ARE THE SAME FOR A OLLAMACHAT CLASS.
    temperature=0.3
    top_p=0.3
    seed=42
    max_tokens=4096
    #NUMBER OF PARAGRAPHS TO ANALYZE PER LLM CALL. LOWER MEANS MORE GRANULAR ANALYSIS.
    money_paragraphs_to_analyze=3
    entity_paragraphs_to_analyze=1
    #INITIALIZE EMPTY LIST WHERE WE WILL STORE THE RESULTS OF THE LLM EXTRACTION PROCESS
    money_json_result_from_llm_extaction = []
    entiy_json_result_from_llm_extaction = []
    
    logger.info(f"================STARTING EXTRACTION PIPELINE================")
    # First, get the filtered list of PDFs
    selected_pdfs = get_filtered_fully_general_series_call_pdfs(input_doc_path)

    # CALL PAPERS DOCUMENTS HAVE MOST OF THE INFORMATION NEEDED. SO WE WILL FOCUS ON THESE. 
    if analyze_only_call_papers:
        if selected_pdfs:        
            # Now, extract the specific title string for each of these PDFs
            extracted_titles = get_series_titles_from_paths(selected_pdfs)

        else:
            logger.error(f"No PDF files matching the criteria found in '{input_doc_path}'.")
            raise ValueError(f"No PDF files matching the criteria found in '{input_doc_path}'.")
    else:
        extracted_titles=selected_pdfs.copy()
    
    for pdf_path, title in extracted_titles.items():
        logger.info(f"EXTRACTION OF MONEY AND ENTITY INFORMATION FROM: {pdf_path.name}. PROJECT ID: {title}")
        
        #CONVERT PDF FILE AT PDF_PATH INTO STRING
        text_output = pdf_converter.convert_to_text(pdf_path)
        #CREATE DOCUMENT OBJECT. IT WILL AUTOMATICALLY SEGMENT THE WHOLE STRING INTO SENTENCES AND PARGRAPHS.
        doc=Document(raw_text=text_output,paragraph_segmentation_mode=paragraph_segmentation_mode)
        #FROM DOCUMENT GET THE PARAGRAPHS (OR SENTENCES) THAT MATCH THE REGEX.
        money_list_sents=doc.get_paras_or_sents_raw_text(reference_depth=reference_depth,regex_pattern=money_regex).copy()
        entity_list_sents=doc.get_paras_or_sents_raw_text(reference_depth=reference_depth,regex_pattern=entity_regex).copy()
        #CREATE LLM OBJECT
        money_llm=LocalLLM(model=money_llm_model,temperature=temperature,top_p=top_p,seed=seed,max_tokens=max_tokens)
        entity_llm=LocalLLM(model=entity_llm_model,temperature=temperature,top_p=top_p,seed=seed,max_tokens=max_tokens)
        #EXTRACT MONEY AND ENTITY INFORMATION FROM THE PARAGRAPHS.
        money_json_llm_extraction = money_llm.extract_information(money_list_sents,max_items_to_analyze_per_call=money_paragraphs_to_analyze, extraction_type='money',reference_depth=reference_depth)
        entity_json_llm_extraction = entity_llm.extract_information(entity_list_sents,max_items_to_analyze_per_call=entity_paragraphs_to_analyze, extraction_type='entity',reference_depth=reference_depth)
        money_json_result_from_llm_extaction.append({title:money_json_llm_extraction})
        entiy_json_result_from_llm_extaction.append({title:entity_json_llm_extraction})

    #STORE THE MONEY RESULT HERE
    with open('money_result.json', 'w') as f:
        json.dump(money_json_result_from_llm_extaction, f,indent=2)
    
    #STORE THE ENTITY RESULT HERE
    with open('entity_result.json', 'w') as f:
        json.dump(entiy_json_result_from_llm_extaction, f,indent=2)

    logger.info(f"================STARTING TRANSFORMATION PIPELINE================")
    #LOAD DF FROM JSON
    money_df=load_df_from_json('money_result.json')
    entity_df=load_df_from_json('entity_result.json')
    #DEFINE MONEY PIPELINE. EACH FUNCTION OF THE PIPELINE MUST HAVE AS ITS FIRST ARGUMENT A PANDAS DATAFRAME
    money_pipeline = Pipeline(functions=[
        ######## VALIDATION 
        # CHECK FOR MISSING DATA
        verify_no_missing_data,
        # VERIFY THAT THERE ARE NO NEGATIVE NUMBERS
        verify_no_negatives,
        # VERIFY THAT THERE ARE NO ''
        verify_no_empty_strings,
        # VERIFY THAT THE VALUE COLUMNS HAS ONLY NUMBERS
        partial(check_numeric_columns, columns_to_check=['value']),
        # VERIFY THAT CONTEXT AND ORIGINAL_SENTENCE AND CURRENCY HAVE ONLY STRINGS
        partial(check_string_columns, columns_to_check=['context', 'original_sentence','currency']),
        # CHECK THAT THE ORIGINAL_SENTENCE HAS AT LEAST ONE NUMBER
        partial(check_columns_satisfy_regex, columns_to_check=['original_sentence'], regex_pattern=r"\d+"),

        ######## TRANSFORM
        # DROP ANY VALUE EQUAL TO ZERO OR LESS IN VALUE. THERE SHOULD NOT BE ANY NEGATIVE VALUES IN VALUE, THOUGH.
        partial(drop_rows_with_non_positive_values, columns_to_check=['value']),
        # DROP ANY VALUE IN CURRENCY THAT DOES NOT SATISFY THE REGEX.
        partial(drop_rows_not_satisfying_regex, columns_to_check=['currency'], regex_pattern=r"^(?:eur|euros|euro|€)$"),
        # DROP ANY VALUE THAT DOES NOT SATISFY THE REGEX IN EITHER ORIGINAL_SENTENCE AND CONTEXT. IF A VALUE SATISFY THE REGEX IN ORIGINAL_SENTENCE OR CONTEXT, WE KEEP IT
        partial(drop_rows_if_no_column_matches_regex, columns_to_check=['original_sentence', 'context'], regex_pattern=r"call|budget|grant|amif"),
        # THERE ARE CASES WHERE IN THE SAME DOCUMENT, WE HAVE DUPLICATED MONEY VALUE THAT MIGHT OR NOT REFER TO THE SAME SEMANTIC CONTEXT. WE TRY AND DROP THOSE DUPLICATES IF THEY ARE SEMANTICALLY SIMILAR ABOVE A CERTAIN THRESHOLD.
        # THE THRESHOLD REFERS TO THE COSINE DISTANCE. THE HIGHER THE THRESHOLD, THE MORE MERGING FREQUENCY. THE LOWER THE THRESHOLD, THE LESS MERGING FREQUENCY.
        partial(remove_semantic_duplicates, groupby_columns=['document_id', 'value'], target_column="context",threshold = 0.5),
        ])
    #DEFINE ENTITY PIPELINE. EACH FUNCTION OF THE PIPELINE MUST HAVE AS ITS FIRST ARGUMENT A PANDAS DATAFRAME
    entity_pipeline = Pipeline(functions=[
        # VALIDATION
        verify_no_missing_data,
        verify_no_negatives,
        verify_no_empty_strings,
        partial(check_string_columns, columns_to_check=['organization_type']),
        partial(verify_list_column_contains_only_ints, columns_to_check=['min_entities']),
        # TRANSFORMATION
        # DELETE REDUNDANT VALUES IN THE INT LISTS IN THE ROWS
        partial(reduce_list_ints_to_unique, target_column='min_entities'),
        # GROUPBY AND STACK
        partial(group_by_document_and_stack_types, target_column='organization_type')
        ])

    # RUN THE PIPELINES.
    money_result_df= money_pipeline.run(money_df)
    entity_result_df= entity_pipeline.run(entity_df)
    # Debugging: Stampa i tipi di dato delle colonne chiave dopo la conversione
    ######## LOAD
    # SAVE THE RESULT TO CSV
    logger.info(f"================STARTING LOAD PIPELINE================")
    logger.info(f"STORING THE RESULT OF THE TRANSFORMATION PIPELINE INTO A CSV")
    #WE ARE GOING TO STORE THE MERGE OF THE RESULTS OF THE TWO TRANSFORMATION PIPELINES INTO A CSV. WE ALSO STORE THE RESULT OF EACH TRANSFROMATION PIPELINE INTO A CSV.
    money_result_df.to_csv('etl_money_result.csv')
    entity_result_df.to_csv('etl_entity_result.csv')
    logger.success(f"ETL PROCESS FINISHED")



    

