import queue
import threading
import time
import random


from Whisperme import Transcriber
from Whisperme import AudioRecorder


# 假设这是你已经实例化的四个不同的模型
class Model1:
    def predict(self, data):
        return f"Model1 processed {data}"

class Model2:
    def predict(self, data):
        return f"Model2 processed {data}"

class Model3:
    def predict(self, data):
        return f"Model3 processed {data}"

class Model4:
    def predict(self, data):
        return f"Model4 processed {data}"

models = [Model1(), Model2(), Model3(), Model4()]

def generate_realtime_data():
    # 生成实时数据的模拟函数
    # 这里生成一个包含5个元素的列表
    return [f"data_{i}" for i in range(5)]

def process_realtime_data(data_list, model, thread_id):
    # 处理实时数据的函数，使用特定的模型
    for data in data_list:
        result = model.predict(data)
        print(f"Thread {thread_id} using {model.__class__.__name__} processing {data}: {result}")

def producer(queue):
    # 模拟实时数据产生过程
    counter = 0
    while True:
        data_list = generate_realtime_data()  # 产生实时数据列表
        queue.put((counter % 4, data_list))  # 将数据列表放入队列，并附带标识符
        counter += 1
        time.sleep(random.uniform(0.5, 1.5))  # 模拟每0.5到1.5秒产生一次数据

def consumer(queue, model, thread_id):
    # 模拟实时数据处理过程
    while True:
        identifier, data_list = queue.get()  # 从队列中获取带有标识的数据列表
        if identifier == thread_id:  # 仅处理与自己标识匹配的数据列表
            process_realtime_data(data_list, model, thread_id)  # 处理数据列表
        queue.task_done()  # 标记任务完成

# 创建一个队列用于存放实时数据
q = queue.Queue()

# 创建一个生产者线程用于产生实时数据
producer_thread = threading.Thread(target=producer, args=(q,))
producer_thread.daemon = True
producer_thread.start()

# # 创建多个消费者线程用于处理实时数据
# num_consumers = 4
# for i in range(num_consumers):
#     consumer_thread = threading.Thread(target=consumer, args=(q, models[i], i))
#     consumer_thread.daemon = True
#     consumer_thread.start()
#
# # 主线程等待所有消费者线程结束
# q.join()
