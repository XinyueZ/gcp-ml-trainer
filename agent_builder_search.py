from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from icecream import ic
from rich.pretty import pprint as pp


def search_sample(
    project_id: str,
    location: str,
    engine_id: str,
    preamble: str,
    search_query: str,
) -> List[discoveryengine.SearchResponse]:
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    client = discoveryengine.SearchServiceClient(client_options=client_options)
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
            model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble=preamble
            ),
            model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                version="stable",
            ),
        ),
    )
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)

    return response


if __name__ == "__main__":
    project_id = "hg-a1050-ai-ft-exp-npr-4291"
    location = "europe-west1"
    app_id = "a4878e61-ab70-4e14-b6cc-3108e5682ca1"  # engine_id
    app_location = "eu"
    aiplatform.init(project=project_id, location=location)
    search_query = (
        """Give me the wine price which is described as 'Fresh cracked peppercorn'."""
    )
    preamble = """Please finally give me only the wine price with proper currency unit without any other additional information or intructions."""
    response = search_sample(
        project_id=project_id,
        location=app_location,
        engine_id=app_id,
        preamble=preamble,
        search_query=search_query,
    )
    ic(response)
    ic(response.summary.summary_text)
