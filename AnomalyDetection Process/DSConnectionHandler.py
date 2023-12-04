
import logging

from botocore.exceptions import ClientError

import LoggerConfig

LOGGER = logging.getLogger('DSConnectionHandler')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER.info('Creating an instance of KnowledgeBase connection handler.')


class Connector:

    def __init__(self, sqs_client, queue_url: str):
        self.sqs_client = sqs_client
        self.queue_url = queue_url

    def receive_messages(self):
        """
        Receive a batch of messages in a single request from an SQS queue.
        :return: The list of Message objects received. These each contain the body
                 of the message and metadata and custom attributes.
        """
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                AttributeNames=['All'],
                MessageAttributeNames=['All'],
                MaxNumberOfMessages=1,
                VisibilityTimeout=5,
                WaitTimeSeconds=0
            )

            if 'Messages' in response:
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle']
                message_body = message['Body']

                # Print or process the received message
                LOGGER.info(f"Received message: {message_body}")

                # Delete the received message from the queue (if needed)
                try:
                    self.sqs_client.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=receipt_handle
                    )
                except ClientError:
                    LOGGER.critical('Could not remove message from SQS queue.')

                return message_body

        except ClientError as error:
            LOGGER.exception("Couldn't receive messages from queue.")
            raise error
