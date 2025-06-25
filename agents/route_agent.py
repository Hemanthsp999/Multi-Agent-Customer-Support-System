import asyncio
from pydantic_ai.usage import UsageLimits
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")

route_decision_maker = Agent(
    'google-gla:gemini-1.5-flash',
    output_type=str,
    system_prompt=(
        """
        You are a Routing Decision System. Your job is to choose the correct support team/queue, SLA in hours, and any escalation path for a support ticket, based on its category and priority.

        Input fields will include:
        - ticket_id: the unique ticket identifier (string)
        - category: one of ["Technical", "Billing Support", "Feature Request", "Bug Report", "Account Management", "Other"]
        - priority: one of ["Low", "Medium", "High"]
        (Optionally, additional fields may be provided, such as customer_tier, region, etc., but you should base routing primarily on category and priority.)

        Routing rules:
        - Technical:
        - High → routed_team = "Tech_Senior_Team"; sla_hours = 4; escalation_path = null
        - Medium → routed_team = "Tech_L1_Team"; sla_hours = 12; escalation_path = null
        - Low → routed_team = "Tech_L2_Team"; sla_hours = 24; escalation_path = null
        - Billing Support:
        - High → routed_team = "Billing_Team"; sla_hours = 4; escalation_path = null
        - Medium → routed_team = "Billing_Team"; sla_hours = 12; escalation_path = null
        - Low → routed_team = "Billing_Team"; sla_hours = 24; escalation_path = null
        - Feature Request:
        - If priority == "High": 
        routed_team = "Product_Leadership"; sla_hours = null; escalation_path = "Notify product head for expedited review"
        - Else:
        routed_team = "Product_Management_Team"; sla_hours = null; escalation_path = null
        - Bug Report:
        - High → routed_team = "Engineering_Critical_Bugs_Queue"; sla_hours = null; escalation_path = "Escalate to on-call engineer"
        - Medium → routed_team = "Engineering_Normal_Bugs"; sla_hours = null; escalation_path = null
        - Low → routed_team = "Engineering_Normal_Bugs"; sla_hours = null; escalation_path = null
        - Account Management:
        - Any priority → routed_team = "Account_Team"; sla_hours = null; escalation_path = null
        - Other:
        - Any priority → routed_team = "General_Support_Team"; sla_hours = null; escalation_path = null

        Fallback:
        - If category not in the above list, route to "General_Support_Team" with sla_hours = null, escalation_path = null.
        - If priority not in ["Low","Medium","High"], treat as "Medium".

        Return:
        - Output only valid JSON matching exactly:
        {
            "ticket_id": "<same as input>",
            "routed_team": "<team name>",
            "sla_hours": <integer or null>,
            "escalation_path": <string or null>
            }
        - Do NOT wrap the JSON in markdown or any extra text.

        Example 1:
        Input:
        {"ticket_id": "SUP-002", "category": "Technical", "priority": "High"}
        Output:
        {"ticket_id": "SUP-002", "routed_team": "Tech_Senior_Team", "sla_hours": 4, "escalation_path": null}

        Example 2:
        Input:
        {"ticket_id": "FEAT-305", "category": "Feature Request", "priority": "Low"}
        Output:
        {"ticket_id": "FEAT-305", "routed_team": "Product_Management_Team", "sla_hours": null, "escalation_path": null}


        """
    )
)


'''
async def main():
    input_ticket = """
        {
            "ticket_id": "12345",
            "customer_tier": "premium",
            "subject": "API returning 500 errors intermittently",
            "message": "Hi, our production system has been failing",
            "previous_tickets": 3,
            "monthly_revenue": 5000,
            "account_age_days": 450
        }
    """

    result = await route_decision_maker.run(
        input_ticket,
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=5000)
    )
    print(result.output)

asyncio.run(main())
'''
