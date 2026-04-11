# AI-Powered Salesforce Stock Signal System

An end-to-end AI-powered decision system that integrates Salesforce, FastAPI, and OpenAI to generate real-time stock trading signals within CRM workflows.

## Architecture

Salesforce (LWC + Apex)
        ↓
Queueable Apex (Async Processing)
        ↓
FastAPI Microservice
        ↓
OpenAI API
        ↓
Response stored back in Salesforce (Opportunity)

- Designed and implemented an AI-powered decision system integrating Salesforce LWC, Apex (Queueable), and a FastAPI microservice to generate real-time stock trading signals within CRM workflows.
- Developed a custom LWC on Opportunity records for user-triggered analysis and real-time visualization of BUY/SELL/HOLD signals with confidence scores and contextual reasoning.
- Engineered asynchronous API integration using Queueable Apex and Named Credentials, ensuring scalable and non-blocking execution within Salesforce governor limits.
- Integrated OpenAI via FastAPI to generate contextual financial insights using market trends, volatility, and news sentiment.
- Implemented dynamic ticker-based analysis and optimized performance using caching, enabling scalable multi-record processing and reducing reliance on manual analysis.
- Tech Stack: Salesforce (Apex, Flow), FastAPI, OpenAI API, Named Credentials, REST Integration, LWC


Note: API responses are cached and refreshed every 15 minutes to optimize server performance on the free tier.

Sample:
![TSLA](https://github.com/user-attachments/assets/46c36379-6756-445e-a33f-cc5fd4ae0619)

Create New Opportunity + Update Opportunity:

https://github.com/user-attachments/assets/a22ffdfe-f903-4330-a797-5b4ff09cedb4

Update Specific Ticket Analysis:

https://github.com/user-attachments/assets/953c926f-85c2-4c29-8d26-0fef87d30bf2






