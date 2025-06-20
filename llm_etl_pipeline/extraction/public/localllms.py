import math
from typing import Any, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import Field, PrivateAttr, validate_call

from llm_etl_pipeline.customized_logger import logger
from llm_etl_pipeline.extraction.internal import _get_template, _when_all_is_lost
from llm_etl_pipeline.extraction.public.parsers.entities import ConsortiumComposition
from llm_etl_pipeline.extraction.public.parsers.monetary_informations import (
    MonetaryInformationList,
)
from llm_etl_pipeline.typings import (
    ExtractionType,
    NonEmptyListStr,
    NonEmptyStr,
    ReferenceDepth,
)


class LocalLLM(ChatOllama):
    """
    A specialized LangChain ChatOllama model designed for local execution,
    incorporating a default system prompt and a Pydantic output parser for
    structured data extraction.

    This class extends `ChatOllama` to provide predefined system instructions
    and handle structured output parsing, streamlining interactions with the
    local LLM for specific extraction tasks.

    Attributes:
        default_system_prompt (Optional[NonEmptyStr]): A system-level prompt
            that sets the context or instructions for the LLM. If not provided
            during initialization, it will be loaded from a template. This attribute
            cannot be changed once populated.
        _parser (PydanticOutputParser): A private attribute that holds the
            PydanticOutputParser instance, configured to parse responses into
            `MonetaryInformationList` objects.
    """

    default_system_prompt: Optional[NonEmptyStr] = Field(default=None)
    _parser: PydanticOutputParser = PrivateAttr()

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initializes the LocalLLM instance, extending ChatOllama.

        This constructor calls the parent `ChatOllama` constructor and
        initializes the Pydantic output parser for `MonetaryInformationList`.

        Args:
            *args: Arbitrary positional arguments passed to the `ChatOllama` parent constructor.
            **kwargs: Arbitrary keyword arguments passed to the `ChatOllama` parent constructor.
        """
        super().__init__(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of an instance, with additional restrictions on specific attributes.

        This method prevents the `default_system_prompt` attribute from being
        reassigned once it has been initially set to a non-empty (truthy) value.
        This ensures the stability of the system prompt after its initial setup.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            ValueError: If attempting to reassign `default_system_prompt`
                        after it has already been populated.
        """
        if name in ["default_system_prompt"]:
            # Prevent default_system_prompt reassignment once populated, to prevent inconsistencies.
            if getattr(self, name, None):
                raise ValueError(
                    f"The attribute `{name}` cannot be changed once populated."
                )
        super().__setattr__(name, value)

    def _set_parser(self, extraction_type: ExtractionType) -> None:
        if extraction_type == "money":
            self._parser = PydanticOutputParser(pydantic_object=MonetaryInformationList)
        else:
            self._parser = PydanticOutputParser(pydantic_object=ConsortiumComposition)

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic hook that runs after model initialization and validation.

        This method ensures that `default_system_prompt` is initialized if it's not
        already set during the model's creation.
        """
        if not self.default_system_prompt:
            logger.info("Default system prompt not set. Initializing from template.")
            self._initialize_default_system_prompt()
        else:
            logger.info("Default system prompt already set.")

    def _initialize_default_system_prompt(self) -> None:
        """
        Renders a system message from a template and sets it as the
        `default_system_prompt` for the LLM.

        This method is called if `default_system_prompt` is not provided
        at initialization. It fetches a "system_message" template, renders
        it with the output language set to 'en', and assigns the result.
        """
        self.default_system_prompt = _get_template(
            "system_message",
            template_type="system",
            template_extension="j2",
        ).render({"output_language": "en"})
        logger.success(f"Default system prompt initialized using the template.")

    def _generate_human_prompt_from_template(
        self,
        extraction_type: ExtractionType = "money",
        reference_depth: ReferenceDepth = "sentences",
    ) -> str:
        """
        Renders a human-facing prompt from a template based on extraction type and context.

        This method constructs a dynamic human message prompt using a template
        ("prompt_message") and parameters like `extraction_type` and `reference_depth`.
        The rendered string is intended to guide the LLM's response for specific tasks.

        Args:
            extraction_type (ExtractionType): The type of information to be extracted (e.g., 'money').
                                              Defaults to 'money'.
            reference_depth (ReferenceDepth): The context from which to extract information
                                                (e.g., 'sentences' or 'paragraphs').
                                                Defaults to 'sentences'.

        Returns:
            str: The rendered human-facing prompt string.
        """
        human_prompt = _get_template(
            "prompt_message",
            template_type="prompt",
            template_extension="j2",
        ).render(
            {"extraction_type": extraction_type, "reference_depth": reference_depth}
        )
        logger.success(
            f"Human prompt generated with extraction_type='{extraction_type}' and reference_depth='{reference_depth}'"
        )
        return human_prompt

    def _create_llm_extraction_pipeline(
        self, human_prompt: NonEmptyStr
    ) -> RunnableLambda:
        """
        Constructs and returns a LangChain Runnable pipeline for LLM-based extraction.

        This pipeline integrates a system prompt, a human prompt, the LLM itself,
        and a Pydantic output parser, with a fallback mechanism for errors.

        Args:
            human_prompt (NonEmptyStr): The human-facing prompt to be included in the pipeline.

        Returns:
            RunnableLambda: A LangChain runnable chain configured for extraction,
                            including a fallback.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.default_system_prompt
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        final_prompt = chat_prompt.partial(
            format_instructions=self._parser.get_format_instructions()
        )

        # TO-DO: Instead of using the same LLM for the original chain, define a new LLM for the fallback chain in the case of error.
        fallback_chain = [RunnableLambda(_when_all_is_lost)]
        # chain = (final_prompt | self | self._parser).with_fallbacks(fallbacks=fallback_chain)
        chain = (final_prompt | self | self._parser).with_fallbacks(
            fallbacks=fallback_chain
        )
        logger.success("LLM extraction pipeline created.")
        return chain

    def _process_document(
        self,
        input_document: NonEmptyListStr,
        llm_extraction_pipeline: RunnableLambda,
        max_items_to_analyze_per_call: int = 6,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Processes a list of text items (e.g., sentences or paragraphs) in batches
        using the provided LLM extraction pipeline.

        This method iterates through the `input_document`, processing chunks of
        `max_items_to_analyze_per_call` items in each batch. It invokes the LLM
        pipeline for each batch and collects the parsed results. It provides
        print statements for progress tracking.

        Args:
            input_document (NonEmptyListStr): A non-empty list of text strings
                                              (sentences or paragraphs) to be processed by the LLM.
            llm_extraction_pipeline (RunnableLambda): The LangChain runnable pipeline
                                                      configured for extraction.
            max_items_to_analyze_per_call (int): The maximum number of text items
                                                 to include in a single LLM call (batch size).
                                                 Defaults to 6.

        Returns:
            dict[str, list[dict[str, Any]]]: A dictionary containing the aggregated
                                             extraction results, typically a list of
                                             parsed JSON objects under the 'amounts' key.
        """
        results = {}
        list_of_json_objects = []
        num_batches = math.ceil(len(input_document) / max_items_to_analyze_per_call)
        logger.info(
            f"Starting text analysis in {num_batches} batches, with {max_items_to_analyze_per_call} items per batch."
        )

        for i in range(num_batches):
            start_index = i * max_items_to_analyze_per_call
            end_index = min(
                (i + 1) * max_items_to_analyze_per_call, len(input_document)
            )
            current_batch_sentences = input_document[start_index:end_index]

            if not current_batch_sentences:
                continue

            document_text = "\n\n".join(current_batch_sentences)

            logger.info(
                f"Processing Batch {i+1}/{num_batches} (Items {start_index+1}-{end_index})"
            )
            llm_results = llm_extraction_pipeline.invoke(document_text)

            if isinstance(llm_results, MonetaryInformationList):
                extraction_result = llm_results.model_dump()["amounts"]
                logger.info(
                    f"Successfully parsed {len(extraction_result)} items from batch {i+1}."
                )
            elif isinstance(llm_results, ConsortiumComposition):
                dump_model = llm_results.model_dump()
                participants = dump_model["participants"]
                min_entities = dump_model["min_entities"]
                extraction_result = [participants, {"min_entities": min_entities}]
                logger.info(
                    f"Successfully parsed {len(extraction_result)} items from batch {i+1}."
                )
            else:
                logger.error(f"Validation failed for batch {i+1}.")
                extraction_result = []
            list_of_json_objects.extend(extraction_result)

        results = {"results": list_of_json_objects}
        logger.success(
            f"Text analysis completed. Total extracted items: {len(list_of_json_objects)}"
        )
        return results

    @validate_call
    def extract_information(
        self,
        list_elem: NonEmptyListStr,
        extraction_type: ExtractionType = "money",
        reference_depth: ReferenceDepth = "sentences",
        max_items_to_analyze_per_call: int = Field(default=4, gt=0),
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Main entry point to perform LLM-based extraction on a list of text elements.

        This method orchestrates the entire extraction process: it generates the
        appropriate human prompt, creates the LLM extraction pipeline, and then
        processes the input list of elements in batches.

        Args:
            list_elem (NonEmptyListStr): A non-empty list of text strings (e.g., sentences or paragraphs)
                                         to be analyzed.
            extraction_type (ExtractionType): The type of information to extract (e.g., 'money', 'entity').
                                              Defaults to 'money'.
            reference_depth (ReferenceDepth): The granular level of text being analyzed
                                                (e.g., 'sentences' or 'paragraphs'). Defaults to 'sentences'.
            max_items_to_analyze_per_call (int): The maximum number of text items
                                                 to include in a single LLM call (batch size).
                                                 Defaults to 4.

        Returns:
            dict[str, list[dict[str, Any]]]: A dictionary containing the final
                                             aggregated extraction results.
        """
        logger.info(
            f"Starting result extraction with {len(list_elem)} elements, extraction_type='{extraction_type}', reference_depth='{reference_depth}', max_items_per_call={max_items_to_analyze_per_call}"
        )
        self._set_parser(extraction_type)
        human_prompt = self._generate_human_prompt_from_template(
            extraction_type, reference_depth
        )
        llm_extraction_pipeline = self._create_llm_extraction_pipeline(human_prompt)
        json_result = self._process_document(
            list_elem, llm_extraction_pipeline, max_items_to_analyze_per_call
        )
        logger.success("Result extraction completed.")
        return json_result
