from textSummarizer.pipelines.stage_01_data_ingestion import DataIngestionPipeline
from textSummarizer.pipelines.stage_02_data_validation import DataValidationPipeline
from textSummarizer.pipelines.stage_03_data_transformation import DataTransformationPipeline
from textSummarizer.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(f"Error occurred in stage {STAGE_NAME}: {e}")
    raise e



STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(f"Error occurred in stage {STAGE_NAME}: {e}")
    raise e



STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")  

except Exception as e:
    logger.exception(f"Error occurred in stage {STAGE_NAME}: {e}")
    raise e
