	IH?m??S@IH?m??S@!IH?m??S@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'IH?m??S@? ??5@1??)?ϏR@I?^?S??r0*	??"??6j@2[
$Iterator::Root::MapAndBatch::Shuffle 4?y?S???!?u??A?N@)4?y?S???1?u??A?N@:Preprocessing2R
Iterator::Root::MapAndBatch?,{؜??!S???I?9@)?,{؜??1S???I?9@:Preprocessing2E
Iterator::Root?n?l???!n?9=?'C@):??l??1Y?uf0)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI ?U?H@Q??jr??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? ??5@? ??5@!? ??5@      ??!       "	??)?ϏR@??)?ϏR@!??)?ϏR@*      ??!       2      ??!       :	?^?S???^?S??!?^?S??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?U?H@y??jr??W@