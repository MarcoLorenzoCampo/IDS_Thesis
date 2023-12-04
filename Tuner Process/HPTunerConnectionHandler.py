import functools
import json
import logging
import Hypertuner

import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from pika.exchange_type import ExchangeType

LOGGER = logging.getLogger('HypertunerConnectionHandler')
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER.info('Creating an instance of KnowledgeBase connection handler.')


class Connector:
    EXCHANGE = 'message'
    EXCHANGE_TYPE = ExchangeType.topic

    CLASS_QUEUE_ROUTING_KEYS = {
        'tuner_queue': 'tuner_key'
    }

    def __init__(self, tuner: Hypertuner, ampq_url):
        # Set values for the pika connection
        self.should_reconnect = False
        self.was_consuming = False

        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._url = ampq_url
        self._consuming = False
        self._prefetch_count = 1
        self._message_number = 0

        self._deliveries = None
        self._acked = None
        self._nacked = None
        self._message_number = None

        self._stopping = False

        # Set reference to the Hypertuner
        self.tuner = tuner

    def connect(self):

        LOGGER.info('Connecting to %s', self._url)
        return AsyncioConnection(
            parameters=pika.URLParameters(self._url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed
        )

    def close_connection(self):
        self._consuming = False
        if self._connection.is_closing or self._connection.is_closed:
            LOGGER.info('Connection is closing or already closed')
        else:
            LOGGER.info('Closing connection')
            self._connection.close()

    def on_connection_open(self, _unused_connection):

        LOGGER.info('Connection opened')
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):

        LOGGER.error('Connection open failed: %s', err)
        self.reconnect()

    def on_connection_closed(self, _unused_connection, reason):

        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            LOGGER.warning('Connection closed, reconnect necessary: %s', reason)
            self.reconnect()

    def reconnect(self):

        self.should_reconnect = True
        self.stop()

    def open_channel(self):

        LOGGER.info('Creating a new channel')
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):

        LOGGER.info('Channel opened')
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.EXCHANGE)

    def add_on_channel_close_callback(self):

        LOGGER.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):

        LOGGER.warning('Channel %i was closed: %s', channel, reason)
        self.close_connection()

    def setup_exchange(self, exchange_name):

        LOGGER.info('Declaring exchange: %s', exchange_name)
        # Note: using functools.partial is not required, it is demonstrating
        # how arbitrary data can be passed to the callback when it is called
        cb = functools.partial(self.on_exchange_declareok, userdata=exchange_name)
        self._channel.exchange_declare(
            exchange=exchange_name,
            exchange_type=self.EXCHANGE_TYPE,
            callback=cb)

    def on_exchange_declareok(self, _unused_frame, userdata):

        LOGGER.info('Exchange declared: %s', userdata)

        for queue_name, routing_key in self.CLASS_QUEUE_ROUTING_KEYS.items():
            self.setup_queue(queue_name)

    def setup_queue(self, queue_name):

        LOGGER.info('Declaring queue %s', queue_name)
        self._channel.queue_declare(queue=queue_name, callback=lambda frame: self.on_queue_declareok(frame, queue_name))

    def on_queue_declareok(self, _unused_frame, userdata):

        queue_name = userdata
        key = self.CLASS_QUEUE_ROUTING_KEYS[queue_name]
        LOGGER.info('Binding %s to %s with %s', self.EXCHANGE, queue_name, key)
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        self._channel.queue_bind(
            queue_name,
            self.EXCHANGE,
            routing_key=key,
            callback=cb)

    def on_bindok(self, _unused_frame, userdata):

        LOGGER.info('Queue bound: %s', userdata)
        self.set_qos(userdata)

    def set_qos(self, userdata):

        def callback(_unused_frame):
            self.on_basic_qos_ok(_unused_frame, userdata)

        self._channel.basic_qos(prefetch_count=self._prefetch_count, callback=callback)

    def on_basic_qos_ok(self, _unused_frame, userdata):

        LOGGER.info('QOS set to: %d', self._prefetch_count)
        self.start_consuming(userdata)

    def start_consuming(self, userdata):

        queue_name = userdata
        LOGGER.info('Issuing consumer related RPC commands')
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(
            queue_name, self.on_message)
        self.was_consuming = True
        self._consuming = True

    def add_on_cancel_callback(self):

        LOGGER.info('Adding consumer cancellation callback')
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):

        LOGGER.info('Consumer was cancelled remotely, shutting down: %r',
                    method_frame)
        if self._channel:
            self._channel.close()

    def on_message(self, _unused_channel, basic_deliver, properties, body):

        LOGGER.info('Received message # %s from %s: %s',
                    basic_deliver.delivery_tag, properties.app_id, body)
        self.acknowledge_message(basic_deliver.delivery_tag)
        result = self.tuner.perform_query(body)

    def acknowledge_message(self, delivery_tag):

        LOGGER.info('Acknowledging message %s', delivery_tag)
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):

        if self._channel:
            LOGGER.info('Sending a Basic.Cancel RPC command to RabbitMQ')
            cb = functools.partial(
                self.on_cancelok, userdata=self._consumer_tag)
            self._channel.basic_cancel(self._consumer_tag, cb)

    def on_cancelok(self, _unused_frame, userdata):

        self._consuming = False
        LOGGER.info(
            'RabbitMQ acknowledged the cancellation of the consumer: %s',
            userdata)
        self.close_channel()

    def close_channel(self):

        LOGGER.info('Closing the channel')
        self._channel.close()

    def run(self):

        self._connection = self.connect()
        self._connection.ioloop.run_forever()

    def stop(self):

        if not self._closing:
            self._closing = True
            LOGGER.info('Stopping')
            if self._consuming:
                self.stop_consuming()
                self._connection.ioloop.run_forever()
            else:
                self._connection.ioloop.stop()
            LOGGER.info('Stopped')

    def publish_message(self, queue_name: str):
        if self._channel is None or not self._channel.is_open:
            return

        routing_key = self.CLASS_QUEUE_ROUTING_KEYS[queue_name]

        hdrs = {'queue': f'{queue_name}', 'key': f'{routing_key}'}
        properties = pika.BasicProperties(app_id='Tuner_publisher',
                                          content_type='application/json',
                                          headers=hdrs)

        json_mess = {
            "select": "*",
            "from": "x_train_l1",
            "where": ""
        }

        message = json.dumps(json_mess, ensure_ascii=False)
        self._channel.basic_publish(self.EXCHANGE, routing_key, message, properties)
        self._message_number += 1
        self._deliveries[self._message_number] = True
        LOGGER.info('Published message # %i', self._message_number)

    def start_publishing(self):
        LOGGER.info('Issuing consumer related RPC commands')
        self.enable_delivery_confirmations()

    def enable_delivery_confirmations(self):
        LOGGER.info('Issuing Confirm.Select RPC command')
        self._channel.confirm_delivery(self.on_delivery_confirmation)

    def on_delivery_confirmation(self, method_frame):

        confirmation_type = method_frame.method.NAME.split('.')[1].lower()
        ack_multiple = method_frame.method.multiple
        delivery_tag = method_frame.method.delivery_tag

        LOGGER.info('Received %s for delivery tag: %i (multiple: %s)',
                    confirmation_type, delivery_tag, ack_multiple)

        if confirmation_type == 'ack':
            self._acked += 1
        elif confirmation_type == 'nack':
            self._nacked += 1

        del self._deliveries[delivery_tag]

        if ack_multiple:
            for tmp_tag in list(self._deliveries.keys()):
                if tmp_tag <= delivery_tag:
                    self._acked += 1
                    del self._deliveries[tmp_tag]
        """
        NOTE: at some point you would check self._deliveries for stale
        entries and decide to attempt re-delivery
        """

        LOGGER.info(
            'Published %i messages, %i have yet to be confirmed, '
            '%i were acked and %i were nacked', self._message_number,
            len(self._deliveries), self._acked, self._nacked)