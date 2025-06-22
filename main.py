import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from agents.ticket_classifier_agent import categorize_ticket_agent, TicketSchema
from agents.priority_agent import get_priority_agent, priority_schema
from agents.route_agent import route_decision_maker
from pydantic import BaseModel


class output_format(BaseModel):
    TicketID: int
    Subject: str
    Message: str
    Category: str
    Priority: str
    RouteTo: str


main_agent = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=output_format,
    system_prompt=(
        """

        You are a Multi-Agent Customer Support System. Your task is to analyze the ticket and assign priority.

        You have access to three tools:
        1. `categorize_ticket_tool` – classifies the ticket into categories like Technical, Billing, Feature Support, etc.
        2. `priority_ticket_tool` – assigns a priority (Low/Medium/High) based on customer tier, previous_tickets, and account_age_days.
        3. `route_decision_tool` – make route path based on the type of subject and category.

        Return:
        - TicketID
        - Subject
        - Message
        - Category
        - Priority
        - RouteTo (routed_team from route_decision_tool)

        You must extract the route decision from the route_decision_tool and insert the 'routed_team' value into RouteTo.

        Do not hallucinate. Use only tool outputs.

        """
    )
)


@main_agent.tool
async def categorize_ticket_tool(ctx: RunContext, input_ticket: str) -> TicketSchema:
    return (await categorize_ticket_agent.run(input_ticket, usage=ctx.usage))


@main_agent.tool
async def priority_ticket_tool(ctx: RunContext, input_ticket: str) -> priority_schema:
    return (await get_priority_agent.run(input_ticket, usage=ctx.usage))


@main_agent.tool
async def route_decision_tool(ctx: RunContext, input_ticket: str) -> str:
    return (await route_decision_maker.run(
        input_ticket,
        usage=ctx.usage
    ))


async def main():
    input_ticket = """

        {

            ticket_id": "SUP-003",
            "customer_tier": "premium",
            "subject": "Feature Request: Bulk export",
            "message": "We need bulk export functionality for our quarterly reports. Currently exporting o"
            "previous_tickets": 5,
            "monthly_revenue": 5000,
            "account_age_days": 400"

        }

    """

    result = await main_agent.run(
        input_ticket,
        usage_limits=UsageLimits(request_limit=15, total_tokens_limit=4000)
    )
    print(result.output, sep="\n")


asyncio.run(main())
