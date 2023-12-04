
import logging

from botocore.exceptions import ClientError

import LoggerConfig

LOGGER = logging.getLogger('DSConnectionHandler')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER.info('Creating an instance of KnowledgeBase connection handler.')


class Connector:

    def __init__(self, sqs, queue_url: str):
        self.receiver_queues = None
        self.sqs = sqs
        self.queue_url = queue_url

    def receive_messages(self, max_number, wait_time):
        """
        Receive a batch of messages in a single request from an SQS queue.

        :param queue: The queue from which to receive messages.
        :param max_number: The maximum number of messages to receive. The actual number
                           of messages received might be less.
        :param wait_time: The maximum time to wait (in seconds) before returning. When
                          this number is greater than zero, long polling is used. This
                          can result in reduced costs and fewer false empty responses.
        :return: The list of Message objects received. These each contain the body
                 of the message and metadata and custom attributes.
        """
        for queue in self.receiver_queues:

            try:
                messages = queue.receive_messages(
                    QueueUrl=self.queue_url,
                    MessageAttributeNames=["All"],
                    MaxNumberOfMessages=max_number,
                    WaitTimeSeconds=wait_time,
                )
                for msg in messages:
                    LOGGER.info("Received message: %s: %s", msg.message_id, msg.body)
            except ClientError as error:
                LOGGER.exception("Couldn't receive messages from queue: %s", queue)
                raise error
            else:
                return messages
