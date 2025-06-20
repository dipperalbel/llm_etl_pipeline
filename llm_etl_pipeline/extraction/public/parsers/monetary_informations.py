from typing import Any

from pydantic import BaseModel, Field, field_validator

from llm_etl_pipeline.typings import NonEmptyStr


class MonetaryInformation(BaseModel):
    """
    Represents a single piece of monetary information extracted from a text.

    Attributes:
        value (float): The numerical value of the monetary amount (e.g., 123.45).
        currency (str): The currency of the amount (e.g., 'euro', 'USD', '€', '$').
        context (str): A brief description of the amount's context in the sentence
                       (e.g., 'product cost', 'monthly salary').
        original_sentence (str): The exact sentence from which the monetary amount was extracted.
    """

    value: float = Field(
        description="The numerical value of the monetary amount, e.g., 123.45",
        frozen=True,
        ge=1,
    )
    currency: NonEmptyStr = Field(
        description="The currency of the amount, e.g., 'euro', 'USD', '€', '$'",
        frozen=True,
    )
    context: NonEmptyStr = Field(
        description="A description of the amount's context in the sentence, e.g., 'product cost', 'monthly salary'",
        frozen=True,
    )
    original_sentence: NonEmptyStr = Field(
        description="The exact sentence from which the monetary amount was extracted.",
        frozen=True,
    )

    @field_validator("original_sentence")
    def _original_sentence_must_contain_digit(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError("Original sentence must contain at least one digit.")
        return v


# Wrapper for the array of monetary object


class MonetaryInformationList(BaseModel):
    """
    A wrapper class to hold a list of extracted MonetaryInformation objects.

    This class facilitates the structured output of multiple monetary amounts
    extracted from a given text for JSON serialization.

    Attributes:
        amounts (list[MonetaryInformation]): A list of objects, each containing
                                            details about a monetary amount found in the text.
    """

    amounts: list[MonetaryInformation] = Field(
        description="A list of objects containing monetary information extracted from the text."
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of an instance, with additional restrictions on specific attributes.

        Prevents `amounts` from being reassigned once they have
        been initially set to a non-empty (truthy) value, to maintain data consistency
        and prevent accidental modification after initial segmentation.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            ValueError: If attempting to reassign 'amounts'
                        after it has already been assigned to a *truthy* value.
        """
        if name in ["amounts"]:
            # Prevent raw_text/paragraphs reassignment once populated, to prevent inconsistencies in analysis.
            if getattr(self, name, None):
                raise ValueError(
                    f"The attribute `{name}` cannot be changed once populated."
                )
        super().__setattr__(name, value)
