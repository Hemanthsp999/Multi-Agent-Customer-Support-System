import asyncio
from pydantic_ai import Agent, RunContext
from evaluation.performance import evaluate_agents
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
        You are an Expert Support Ticket Analyzer and Router. Your mission is to process incoming support tickets through a systematic multi-step analysis to determine the most appropriate handling approach.

        ## Your Workflow:
        1. **CATEGORIZE**: Use categorize_ticket_tool to classify the ticket type
        2. **PRIORITIZE**: Use priority_ticket_tool to determine urgency level
        3. **ROUTE**: Use route_decision_tool with the ticket_id, category, and priority to determine the best team assignment

        ## Available Tools:
        - `categorize_ticket_tool(input_ticket)`: Analyzes ticket content and returns category classification
        - `priority_ticket_tool(input_ticket)`: Evaluates customer data and returns priority level
        - `route_decision_tool(ticket_id, category, priority)`: Determines optimal team routing based on classification

        ## Critical Instructions:
        - **ALWAYS** call tools in the exact order: categorize → prioritize → route
        - **EXTRACT** ticket_id, subject, and message from the input JSON
        - **USE** the category output from categorize_ticket_tool
        - **USE** the priority output from priority_ticket_tool
        - **PARSE** the JSON response from route_decision_tool and extract the "routed_team" value for RouteTo
        - **NEVER** make assumptions or hallucinate - only use actual tool outputs

        ## Expected Output Format (output should only be generated from tools):
        - TicketID: Extract from input ticket_id field
        - Subject: Extract from input subject field
        - Message: Extract from input message field
        - Category: Use exact output from categorize_ticket_tool
        - Priority: Use exact output from priority_ticket_tool
        - RouteTo: Extract "routed_team" from route_decision_tool response, Don't give answers other than `route_decision_maker`.

        ## Error Handling:
        - If any tool fails, note the failure but continue with available data
        - If routing fails, default RouteTo to "General_Support_Team"
        - If Input ticket not consists of these fields:
            "ticket_id",
            "customer_tier",
            "subject",
            "message",
            "previous_tickets",
            "monthly_revenue",
            "account_age_days"
        then return with message "Give valid ticket"

        Remember: You are the orchestrator ensuring each ticket gets proper analysis and routing through the specialized agent tools.
        """
    )
)


@main_agent.tool
async def categorize_ticket_tool(ctx: RunContext, input_ticket: list[str]) -> TicketSchema:
    return (await categorize_ticket_agent.run(input_ticket, usage=ctx.usage))


@main_agent.tool
async def priority_ticket_tool(ctx: RunContext, input_ticket: list[str]) -> priority_schema:
    return (await get_priority_agent.run(input_ticket, usage=ctx.usage))


@main_agent.tool
async def route_decision_tool(ctx: RunContext, ticket: TicketSchema, priority: priority_schema) -> str:
    """Routes the ticket to appropriate team based on category and priority"""

    route = {
        "ticket_id": ticket.TicketID,
        "Category": ticket.Category,
        "Priority": priority.priority
    }

    return (await route_decision_maker.run(
        route,
        usage=ctx.usage
    ))


async def main():

    input_ticket = """

        {
            "ticket_id": "SUP-001",
            "customer_tier": "free",
            "subject": "This product is completely broken!!!",
            "message": "Nothing works! I can't even log in. This is the worst software I've ever used. I'm",
            "previous_tickets": 0,
            "monthly_revenue": 0,
            "account_age_days": 2
        },
        """

    test_set = [
        {
            "input": {

                "ticket_id": "SUP-001",
                "customer_tier": "free",
                "subject": "This product is completely broken!!!",
                "message": "Nothing works! I can't even log in. This is the worst software I've ever used. I'm",
                "previous_tickets": 0,
                "monthly_revenue": 0,
                "account_age_days": 2

            },
            "expected": {
                "category": "Account Management",
                "priority": "Low",
                "routed_team": "Account_Team"
            }
        },

        # Add more test cases as needed...
    ]

    result = await main_agent.run(
        input_ticket,
        usage_limits=UsageLimits(request_limit=15, total_tokens_limit=10000)
    )
    print(result.output)


asyncio.run(main())
