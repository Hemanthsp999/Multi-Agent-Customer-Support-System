import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic import BaseModel
from dotenv import load_dotenv
import os


load_dotenv()
os.getenv("GOOGLE_API_KEY")


class priority_schema(BaseModel):
    ticketId: int
    customer_tire: str
    priority: int


class TicketSchema(BaseModel):
    customer_tire: str
    account_age_days: int


get_ticket_info = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=list[str],
    system_prompt=(
        """
        You are a Ticket Scrapper. Read the ticket throughly and extract contents only from `customer_tire` and `account_age_days`.
        Also store the values insdie {TicketSchema} and return the {TicketSchema}.
        """
    )
)

get_agent = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=priority_schema,
    system_prompt=(
        """
        You are a Priority Assigner. Read the contents from {TicketSchema} of `customer_tire` and `account_age_days`.
        If `customer_tire` consists of:
            1. free -> Low.
            2. premium -> Medium.
            3. enterprise -> High.

        And `account_age_days` is greater than 100, then the customer is regular one.

        By checking both customer_tire and account_age_days give priority range(1-5) and return {priority_schema}.
        """
    )
)


@get_agent.tool
async def extract_ticket_info(ctx: RunContext[None], ticket_input: list[str]) -> priority_schema:
    try:
        r = await get_ticket_info.run(
            ticket_input,
            usage=ctx.usage
        )

        return r.output

    except Exception as e:
        return f"Error: {str(e)}"


async def main():
    input_ticket = """
    {
        "ticket_id": 12345,
        "customer_tier": "enterprise",
        "account_age_days": 450,
        "message": "Our systems are failing randomly."
    }
    """

    result = await get_agent.run(
        input_ticket,
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=500)
    )
    print(result.output)

asyncio.run(main())

