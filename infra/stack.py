"""
Invoice Parser — AWS CDK Stack

Resources:
  - S3 bucket (uploads with 90-day lifecycle)
  - DynamoDB table (on-demand, GSI for status queries)
  - ECR repository
  - ECS Fargate cluster, task definition, service
  - ALB (HTTP :80 -> ECS :8000)
  - IAM roles for S3, DynamoDB, Bedrock access
"""

from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    CfnOutput,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_logs as logs,
    aws_elasticloadbalancingv2 as elbv2,
)
from constructs import Construct


class InvoiceParserStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account = Stack.of(self).account
        region = Stack.of(self).region

        # =====================================================================
        # S3 Bucket
        # =====================================================================
        bucket = s3.Bucket(
            self,
            "InvoiceBucket",
            bucket_name=f"invoice-parser-{account}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-uploads-90d",
                    prefix="uploads/",
                    expiration=Duration.days(90),
                ),
            ],
        )

        # =====================================================================
        # DynamoDB Table
        # =====================================================================
        table = dynamodb.Table(
            self,
            "InvoiceTable",
            table_name="invoice-parser",
            partition_key=dynamodb.Attribute(
                name="pk", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sk", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # GSI: query all verifications by status
        table.add_global_secondary_index(
            index_name="by-status",
            partition_key=dynamodb.Attribute(
                name="sk", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="pk", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # =====================================================================
        # ECR Repository
        # =====================================================================
        repo = ecr.Repository(
            self,
            "InvoiceECR",
            repository_name="invoice-parser",
            removal_policy=RemovalPolicy.RETAIN,
        )

        # =====================================================================
        # VPC (default)
        # =====================================================================
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # =====================================================================
        # ECS Cluster
        # =====================================================================
        cluster = ecs.Cluster(
            self,
            "InvoiceCluster",
            cluster_name="invoice-parser",
            vpc=vpc,
        )

        # =====================================================================
        # Task Role — S3, DynamoDB, Bedrock
        # =====================================================================
        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        # S3 read/write
        bucket.grant_read_write(task_role)

        # DynamoDB CRUD
        table.grant_read_write_data(task_role)

        # Bedrock InvokeModel
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:{region}::foundation-model/*",
                    f"arn:aws:bedrock:{region}:{account}:inference-profile/*",
                ],
            )
        )

        # =====================================================================
        # CloudWatch Log Group
        # =====================================================================
        log_group = logs.LogGroup(
            self,
            "InvoiceLogs",
            log_group_name="/ecs/invoice-parser",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # =====================================================================
        # ECS Task Definition (Fargate)
        # =====================================================================
        task_def = ecs.FargateTaskDefinition(
            self,
            "InvoiceTaskDef",
            cpu=2048,
            memory_limit_mib=4096,
            task_role=task_role,
            ephemeral_storage_gib=30,
        )

        container = task_def.add_container(
            "invoice-parser",
            image=ecs.ContainerImage.from_ecr_repository(repo, tag="latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="invoice-parser",
                log_group=log_group,
            ),
            environment={
                "STORAGE_BACKEND": "aws",
                "S3_BUCKET": bucket.bucket_name,
                "DYNAMO_TABLE": "invoice-parser",
                "AWS_REGION": region,
            },
        )

        container.add_port_mappings(
            ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP)
        )

        # =====================================================================
        # ALB
        # =====================================================================
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "InvoiceALB",
            vpc=vpc,
            internet_facing=True,
        )

        listener = alb.add_listener(
            "HttpListener",
            port=80,
            open=True,
        )

        # =====================================================================
        # ECS Service (Fargate)
        # =====================================================================
        service = ecs.FargateService(
            self,
            "InvoiceService",
            cluster=cluster,
            task_definition=task_def,
            desired_count=1,
            assign_public_ip=True,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        listener.add_targets(
            "EcsTarget",
            port=8000,
            targets=[service],
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # Allow ALB -> ECS traffic
        service.connections.allow_from(
            alb, ec2.Port.tcp(8000), "ALB to ECS"
        )

        # =====================================================================
        # Outputs
        # =====================================================================
        CfnOutput(self, "AlbDns", value=alb.load_balancer_dns_name)
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "TableName", value=table.table_name)
        CfnOutput(self, "EcrRepoUri", value=repo.repository_uri)
