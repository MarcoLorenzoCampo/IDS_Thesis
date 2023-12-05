
import logging
import random
import string

from botocore.exceptions import ClientError

import LoggerConfig

LOGGER = logging.getLogger('DSConnectionHandler')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER.info('Creating an instance of KnowledgeBase connection handler.')


class Connector:

    def __init__(self, sqs_client, sqs_resource, queue_url: str):
        self.output_queue = None
        self.output_queue_name = 'forward-metrics.fifo'
        self.sqs_resource = sqs_resource
        self.sqs_client = sqs_client
        self.queue_url = queue_url

        self.msg_counter = 1

        self.__setup()

    def __setup(self):
        attributes = {
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true'
        }
        self.__create_queue(queue_name=self.output_queue_name, attributes=attributes)

    def __create_queue(self, queue_name: str, attributes: dict = None):
        """
        Creates an Amazon SQS queue.

        :param queue_name: The name of the queue. This is part of the URL assigned to the queue.
        :param attributes: The attributes of the queue, such as maximum message size or
                           whether it's a FIFO queue.
        :return: A Queue object that contains metadata about the queue and that can be used
                 to perform queue operations like sending and receiving messages.
        """

        if not attributes:
            attributes = {}

        try:
            queue = self.sqs_resource.create_queue(
                QueueName=queue_name,
                Attributes=attributes
            )
            self.output_queue = queue

            LOGGER.info("Created FIFO queue '%s' with URL=%s", queue_name, queue.url)
        except ClientError as error:
            LOGGER.exception("Couldn't create queue named '%s'.", queue_name)
            raise error
        else:
            return queue

    def send_message(self, message_body, message_attributes=None):
        """
        Send a message to an Amazon SQS queue.

        :param message_body: The body text of the message.
        :param message_attributes: Custom attributes of the message. These are key-value
                                   pairs that can be whatever you want.
        :return: The response from SQS that contains the assigned message ID.
        """
        if not message_attributes:
            message_attributes = {}

        deduplication_id = self.__gen_random_id()
        try:
            response = self.output_queue.send_message(
                MessageBody=message_body,
                MessageGroupId=self.output_queue_name,
                MessageAttributes=message_attributes,
                MessageDeduplicationId=deduplication_id,
            )
            LOGGER.info(f"Sent message #{self.msg_counter}: '{message_body}' to '{self.output_queue_name}'.")
            self.msg_counter += 1

        except ClientError as error:
            LOGGER.exception("Send message failed: %s", message_body)
            raise error
        else:
            return response

    @staticmethod
    def __gen_random_id():
        x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(20))
        return str(x)

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

                LOGGER.info(f"Received message: {message_body}")

                # Delete the received message from the queue
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
