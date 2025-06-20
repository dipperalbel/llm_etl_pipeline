from typing import Any

from pydantic import BaseModel, Field, field_validator

from llm_etl_pipeline.typings import NonEmptyStr, NonZeroInt


class ConsortiumParticipant(BaseModel):
    """
    Represents a single participant in the consortium.

    Attributes:
        organization_type (NonEmptyStr): The type of organization
                                         (e.g., 'non-profit', 'university', 'SME', 'public body').
    """

    organization_type: NonEmptyStr = Field(
        description="The type of organization (e.g., 'non-profit', 'university', 'SME', 'public body')",
        frozen=True,
    )


class ConsortiumComposition(BaseModel):
    """
    Represents the consortium composition for a call for proposals.

    Attributes:
        participants (List[ConsortiumParticipant]): A list of participants.
        min_entities list[NonZeroInt]: A list of minimum number of entities.
    """

    min_entities: list[NonZeroInt] = Field(
        # Explicitly set the default to None
        description="The minimum number of entities. The value(s) are to be found on the row Entities.",
        frozen=True,
    )
    participants: list[ConsortiumParticipant] = Field(
        description="A list of participants."
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of an instance, with additional restrictions on specific attributes.

        Prevents `participants` from being reassigned once it has been initially
        set to a non-empty (truthy) value, to maintain data consistency and prevent
        accidental modification after initial analysis.

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            ValueError: If attempting to reassign 'participants' after it has
                        already been assigned to a *truthy* value.
        """
        if name == "participants":
            # Check if the attribute already has a truthy value (i.e., not None and not an empty list/etc.)
            # Using hasattr and getattr avoids issues if the attribute hasn't been set at all yet.
            if hasattr(self, name) and getattr(self, name):
                raise ValueError(
                    f"The attribute `{name}` cannot be changed once populated."
                )
        super().__setattr__(name, value)
