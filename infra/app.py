#!/usr/bin/env python3
"""CDK app entry point for Invoice Parser infrastructure."""

import aws_cdk as cdk

from stack import InvoiceParserStack

app = cdk.App()

InvoiceParserStack(
    app,
    "InvoiceParserStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    description="Invoice Parser - FastAPI on ECS Fargate with S3, DynamoDB, Bedrock",
)

app.synth()
