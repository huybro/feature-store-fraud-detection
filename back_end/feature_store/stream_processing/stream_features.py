from pyflink.datastream.functions import ProcessWindowFunction, CoProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common import Types
from typing import Iterable

class TxnCountLast10Min(ProcessWindowFunction):
    def process(self, key, context, elements: Iterable[dict]):
        count = len(list(elements))
        for element in elements:
            yield {**element, 'txn_count_last_10_min': str(count)}

class AvgAmtLast1Hour(ProcessWindowFunction):
    def process(self, key, context, elements: Iterable[dict]):
        amounts = [float(e['amount']) for e in elements]
        avg = sum(amounts) / len(amounts)
        for element in elements:
            yield {**element, 'avg_amt_last_1_hour': str(avg)}

class CombineTxnAndAvg(CoProcessFunction):
    def open(self, ctx: RuntimeContext):
        self.txn_count_state = ctx.get_state(ValueStateDescriptor("txn_count", Types.MAP(Types.STRING(), Types.STRING())))
        self.avg_amt_state = ctx.get_state(ValueStateDescriptor("avg_amt", Types.MAP(Types.STRING(), Types.STRING())))

    def process_element1(self, value, ctx):
        self.txn_count_state.update(value)
        if (avg := self.avg_amt_state.value()):
            yield {**value, **avg}

    def process_element2(self, value, ctx):
        self.avg_amt_state.update(value)
        if (txn := self.txn_count_state.value()):
            yield {**txn, **value}

class FinalJoiner(CoProcessFunction):
    def open(self, ctx: RuntimeContext):
        self.distance_state = ctx.get_state(ValueStateDescriptor("distance", Types.MAP(Types.STRING(), Types.STRING())))
        self.stats_state = ctx.get_state(ValueStateDescriptor("stats", Types.MAP(Types.STRING(), Types.STRING())))

    def process_element1(self, distance, ctx):
        self.distance_state.update(distance)
        if (stats := self.stats_state.value()):
            yield {**distance, **stats}

    def process_element2(self, stats, ctx):
        self.stats_state.update(stats)
        if (distance := self.distance_state.value()):
            yield {**distance, **stats}