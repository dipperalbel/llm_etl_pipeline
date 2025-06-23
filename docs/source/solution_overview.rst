Solution Overview
============

During the analysis of the AMIF grant project PDFs, it became clear that they do not contain project-specific **Technology Readiness Level (TRL)** information. While TRL is a standard concept in Horizon Europe projects, the documents only provided generic definitions (TRL 1 to 9) without practical application to the projects themselves. Our attempts to extract any TRL data, even using advanced LLM AI models like Gemini, were unsuccessful and often resulted in hallucinatory outputs. This outcome is consistent with our understanding from both internet research and Gemini, which suggests that TRL — a metric primarily focused on technological maturity — is not typically indicated in AMIF grant projects, given their social rather than technological nature.

Consequently, the solution prioritizes the extraction of the following data points:

* **Budget Information:** This includes detailed proposal budget and grant amounts per project, which are consistently and clearly documented within the **"call for proposal" PDFs**.
* **Organization Details:** We also aimed to extract the number and type of organizations involved in the grants. However, this proved challenging due to the ambiguity and lack of clear definitions in the task documentation. Without a precise understanding of what "number and type of organization" specifically entails within the grant context, we couldn't fully implement this extraction or clarify it through further inquiry.

Given that many of the provided PDFs were found to be templates or contained minimal additional data relevant to the extraction goals, the core focus of this solution was directed exclusively towards processing the "call for proposal" PDFs, as they proved to be the most valuable source of actionable information.