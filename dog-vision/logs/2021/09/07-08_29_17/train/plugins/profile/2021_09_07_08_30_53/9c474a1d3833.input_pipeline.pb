	??ǚ-O@??ǚ-O@!??ǚ-O@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??ǚ-O@ǜg?K?	@1A??! ?L@I;S輆??r0*	!?rh?I`@2R
Iterator::Root::MapAndBatch+?~NA??!??|??F@)+?~NA??1??|??F@:Preprocessing2E
Iterator::Root?c???!?p?m?U@)??ދ/ګ?1Z?d??D@:Preprocessing2[
$Iterator::Root::MapAndBatch::ShuffleӢ>?6??!?{\??)@)Ӣ>?6??1?{\??)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?_?e?@Q???5W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ǜg?K?	@ǜg?K?	@!ǜg?K?	@      ??!       "	A??! ?L@A??! ?L@!A??! ?L@*      ??!       2      ??!       :	;S輆??;S輆??!;S輆??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?_?e?@y???5W@