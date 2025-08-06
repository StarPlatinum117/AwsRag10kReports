FROM public.ecr.aws/lambda/python:3.12

# Copy all code + dependencies into Lambda's working dir.
COPY deployment/ ${LAMBDA_TASK_ROOT}/

# In case packages need to be reinstalled, uncomment the line below.
# RUN pip3 install -r ${LAMBDA_TASK_ROORT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the Lambda handler (modile.function).
CMD ["aws.lambda_handler.lambda_handler"]
