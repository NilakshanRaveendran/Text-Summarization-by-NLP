import os
from textSummarizer.logging import logger
from textSummarizer.entity import (DataValidationConfig)

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_required_files(self) -> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "samsum_dataset"))

            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as status_file:
                        status_file.write(f"Required file: {file} is Not present\nValidation Status: {validation_status}\n ")

                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as status_file:
                        status_file.write(f"Required file: {file} is present\nValidation Status: {validation_status}\n")

            return validation_status

        except Exception as e:
            logger.exception("An error occurred during required files validation")
            raise e 