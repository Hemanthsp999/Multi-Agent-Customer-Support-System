from pydantic_ai import Agent, RunContext
# import asyncio
from pydantic_ai.usage import UsageLimits
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")


class TicketSchema(BaseModel):
    TicketID: int
    Subject: str
    Message: str
    Category: str


ticket_info_collector = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=list[str],
    system_prompt=(
        """
            You are a simple Ticket Info Scrapper. Extract the relevant information about subject and message from the given {ticket_input}.

            NOTE: Don't give answers that is not present in the {ticket_input}

            The Subject and Message should be:
            1. The Subject should be minimum in length and more weightage for words.
            2. The Message should be well detailed and structured.
            """
    )
)

categorize_ticket_agent = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=TicketSchema,
    system_prompt=(
        """
            You are Ticket_Categorize System. Use `get_ticket_info()` tool to extract the information from {ticket_input}.

            NOTE:
            * Don't give answers other than {ticket_input} info.

            1. Based on Subject you need to categorize ticket. The Ticket can be categorized into several types: Technical, Billing Support,
            Feature Support, Bug Report, etc. Use reasonable category and return {TicketSchema}.
            """
    )
)


@categorize_ticket_agent.tool
async def get_ticket(ctx: RunContext[None], ticket) -> TicketSchema:
    try:
        r = await ticket_info_collector.run(
            ticket,
            usage=ctx.usage
        )

        return r.output

    except Exception as e:
        return f"Error: {str(e)}"


'''
async def main():
    input_ticket = """
        {
            "ticket_id": "12345",
            "customer_tier": "premium",  # free/premium/enterprise
            "subject": "API returning 500 errors intermittently",
            "message": "Hi, our production system has been failing",
            "previous_tickets": 3,
            "monthly_revenue": 5000,
            "account_age_days": 450
        }
    """

    result = await categorize_ticket_agent.run(
        input_ticket,
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=500)
    )
    print("Subject: ", {result.output.Subject})
    print("Message: ", {result.output.Message})
    print("Category: ", {result.output.Category})
    print(result.output)

asyncio.run(main())
'''
