import logging
import random
import string

from botocore.exceptions import ClientError

import LoggerConfig

logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER = logging.getLogger('KnowledgeBase')


class Connector:
    prefix = '-update'
    suffix = '.fifo'

    msg_counter = 1

    def __init__(self, sqs):
        self.sqs = sqs

        self.queue_names = [
            'tuner'+self.prefix+self.suffix,
            'detection-system'+self.prefix+self.suffix]

        self.queues = {}

        self.setup()

    def setup(self):

        attributes = {
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true'
        }

        for queue_name in self.queue_names:
            self.create_queue(queue_name=queue_name, attributes=attributes)

    def create_queue(self, queue_name: str, attributes: dict = None):
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
            queue = self.sqs.create_queue(
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

    def fanout_send_message(self, message_body, attributes=None):

        for queue_name in self.queue_names:
            queue = self.queues[queue_name]
            self.send_message(queue, queue_name, message_body, attributes)

    def send_message(self, queue, queue_name, message_body, message_attributes=None):
        """
        Send a message to an Amazon SQS queue.

        :param queue_name:
        :param queue: The queue that receives the message.
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
            LOGGER.info("Sent message #%d: '%s' to '%s'.", self.msg_counter, message_body, queue_name)
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

    def get_queues(self, prefix=None):
        """
        Gets a list of SQS queues. When a prefix is specified, only queues with names
        that start with the prefix are returned.

        :param prefix: The prefix used to restrict the list of returned queues.
        :return: A list of Queue objects.
        """
        if prefix:
            queue_iter = self.sqs.queue_names.filter(QueueNamePrefix=prefix)
        else:
            queue_iter = self.sqs.receiver_queues.all()

        queues = list(queue_iter)
        if queues:
            LOGGER.info("Got queues: %s", ", ".join([q.url for q in queues]))
        else:
            LOGGER.warning("No queues found.")

        return self.queues

    def close(self):
        """
        Removes an SQS queue. When run against an AWS account, it can take up to
        60 seconds before the queue is actually deleted.

        :param queue: The queue to delete.
        :return: None
        """
        for queue in self.queues.values():
            try:
                queue.delete()
                LOGGER.info("Deleted queue with URL=%s.", queue.url)
            except ClientError as error:
                LOGGER.exception("Couldn't delete queue with URL=%s!", queue.url)
                raise error
