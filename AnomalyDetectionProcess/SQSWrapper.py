import logging
import os
import random
import string
from typing import List

from KBProcess import LoggerConfig
from botocore.exceptions import ClientError


logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)


class Connector:
    """
    This class is used to interact with Amazon SQS. It is responsible for creating queues, sending messages
    and receiving messages, deleting the queues, and more. Acts as a wrapper around Amazon SQS.
    """

    def __init__(self, sqs_client=None, sqs_resource=None, queue_urls: List[str] = None, queue_names: List[str] = None):
        """
        :param sqs_client: client instance to read messages from Amazon SQS.
        :param sqs_resource: resource to interact with Amazon SQS.
        :param queue_urls: queue urls to fetch messages from.
        :param queue_names: queue names to be created.
        """
        self.queues = {}
        self.queue_names = queue_names
        self.sqs_resource = sqs_resource
        self.sqs_client = sqs_client
        self.queue_urls = queue_urls

        self.msg_counter = 1

        if queue_names is not None:
            self.__create_queues()

    def __create_queues(self):
        LOGGER.info('Creating a queue for each queue name provided.')
        for queue_name in self.queue_names:
            attributes = {
                'FifoQueue': 'true',
                'ContentBasedDeduplication': 'true'
            }
            self.__create_queue(queue_name=queue_name, attributes=attributes)

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
            self.queues[queue_name] = queue

            LOGGER.info("Created FIFO queue '%s' with URL=%s", queue_name, queue.url)
        except ClientError as error:
            LOGGER.exception("Couldn't create queue named '%s'.", queue_name)
            raise error
        else:
            return queue

    def send_message_to_queues(self, message_body, attributes=None):
        """
        Proxy method that sends a message to all the queues in the wrapper.
        :param message_body: String or dictionary that contains the body of the message.
        :param attributes: Optional attributes of the message. These are key-value pairs that can be whatever you want.
        """
        for queue_name in self.queue_names:
            queue = self.queues[queue_name]
            self.send_message(queue, queue_name, message_body, attributes)

    def send_message(self, queue, queue_name, message_body, message_attributes=None):
        """
        Send a message to an Amazon SQS queue.
        :param queue:
        :param queue_name:
        :param message_body: The body text of the message.
        :param message_attributes: Custom attributes of the message. These are key-value
                                   pairs that can be whatever you want.
        :return: The response from SQS that contains the assigned message ID.
        """
        if not message_attributes:
            message_attributes = {}

        deduplication_id = self.__gen_random_id()
        try:
            response = queue.send_message(
                MessageBody=message_body,
                MessageGroupId=queue_name,
                MessageAttributes=message_attributes,
                MessageDeduplicationId=deduplication_id,
            )
            LOGGER.info(f"Sent message #{self.msg_counter}: '{message_body}' to '{queue_name}'.")
            LOGGER.info(f"Sent message with message_id: {response['MessageId']}")
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
        for queue_url in self.queue_urls:
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl=queue_url,
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
                            QueueUrl=queue_url,
                            ReceiptHandle=receipt_handle
                        )
                    except ClientError:
                        LOGGER.critical('Could not remove message from SQS queue.')

                    return message_body

            except ClientError as error:
                LOGGER.exception("Couldn't receive messages from queue.")
                raise error

    def close(self):
        """
        Removes a SQS queue. When run against an AWS account, it can take up to
        60 seconds before the queue is actually deleted.

        :return: None
        """
        for queue in self.queues.values():
            try:
                queue.delete()
                LOGGER.info("Deleted queue with URL=%s.", queue.url)
            except ClientError as error:
                LOGGER.exception("Couldn't delete queue with URL=%s!", queue.url)
                raise error
