#!/bin/bash
set -e

# Invoice Parser — AWS Deployment (no local Docker needed)
# Usage: ./deploy.sh [region]

REGION="${1:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
STACK_NAME="InvoiceParserStack"
BUCKET="invoice-parser-${ACCOUNT}"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/invoice-parser"

echo "========================================"
echo "  Invoice Parser — Deploy to AWS"
echo "========================================"
echo "  Region:  ${REGION}"
echo "  Account: ${ACCOUNT}"
echo "  Stack:   ${STACK_NAME}"
echo "========================================"

# Get default VPC and subnets
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --region ${REGION} --query 'Vpcs[0].VpcId' --output text)
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=${VPC_ID}" "Name=default-for-az,Values=true" \
  --region ${REGION} --query 'Subnets[*].SubnetId' --output text | tr '\t' ',')

echo "  VPC:     ${VPC_ID}"
echo "  Subnets: ${SUBNET_IDS}"
echo "========================================"

# Step 1: Deploy CloudFormation stack
echo ""
echo "[1/4] Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file infra/template.yaml \
  --stack-name ${STACK_NAME} \
  --parameter-overrides VpcId=${VPC_ID} SubnetIds=${SUBNET_IDS} \
  --capabilities CAPABILITY_IAM \
  --region ${REGION} \
  --no-fail-on-empty-changeset

echo "  Stack deployed."

# Step 2: Package source code and upload to S3
echo ""
echo "[2/4] Packaging source and uploading to S3..."
TMPZIP=$(mktemp /tmp/invoice-source-XXXXXX.zip)
cd "$(dirname "$0")"
zip -r "${TMPZIP}" Dockerfile local_agent/ -x "local_agent/__pycache__/*" "local_agent/.pytest_cache/*" "local_agent/test_*" "local_agent/data/traces/*" "local_agent/data/verifications/*" "local_agent/data/feedback/*" "local_agent/data/metrics/*" "local_agent/data/patterns/*" > /dev/null
aws s3 cp "${TMPZIP}" "s3://${BUCKET}/build/source.zip" --region ${REGION}
rm -f "${TMPZIP}"
echo "  Source uploaded to s3://${BUCKET}/build/source.zip"

# Step 3: Build Docker image via CodeBuild
echo ""
echo "[3/4] Building Docker image on AWS (CodeBuild)..."
BUILD_ID=$(aws codebuild start-build \
  --project-name invoice-parser-build \
  --region ${REGION} \
  --environment-variables-override \
    "name=SOURCE_BUCKET,value=${BUCKET}" \
  --query 'build.id' --output text)
echo "  Build started: ${BUILD_ID}"
echo "  Waiting for build to complete..."

# Poll until complete
while true; do
  STATUS=$(aws codebuild batch-get-builds --ids ${BUILD_ID} \
    --region ${REGION} --query 'builds[0].buildStatus' --output text)
  if [ "${STATUS}" = "SUCCEEDED" ]; then
    echo "  Build SUCCEEDED"
    break
  elif [ "${STATUS}" = "FAILED" ] || [ "${STATUS}" = "FAULT" ] || [ "${STATUS}" = "STOPPED" ]; then
    echo "  Build ${STATUS}. Check CodeBuild logs."
    aws codebuild batch-get-builds --ids ${BUILD_ID} \
      --region ${REGION} --query 'builds[0].phases[-1].contexts' --output text
    exit 1
  fi
  printf "."
  sleep 10
done

# Step 4: Force new ECS deployment
echo ""
echo "[4/4] Updating ECS service with new image..."
aws ecs update-service \
  --cluster invoice-parser \
  --service invoice-parser \
  --force-new-deployment \
  --region ${REGION} > /dev/null

echo ""
echo "========================================"
echo "  Deploy complete!"
echo ""
echo "  URL:"
aws cloudformation describe-stacks \
  --stack-name ${STACK_NAME} \
  --region ${REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`ALBURL`].OutputValue' \
  --output text
echo "========================================"
