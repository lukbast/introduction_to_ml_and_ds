	??y?*N@??y?*N@!??y?*N@	2]Gڳ??2]Gڳ??!2]Gڳ??"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??y?*N@??.	@1GT?n.?K@I%y?????Y??a?'??r0*	)\????T@2R
Iterator::Root::MapAndBatch{?????!??W??hI@){?????1??W??hI@:Preprocessing2E
Iterator::Root??&????!g8L??U@)??o?㆟?1'r??nB@:Preprocessing2[
$Iterator::Root::MapAndBatch::Shuffle%t??Y??!?<????(@)%t??Y??1?<????(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no92]Gڳ??I??'@Q?h???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??.	@??.	@!??.	@      ??!       "	GT?n.?K@GT?n.?K@!GT?n.?K@*      ??!       2      ??!       :	%y?????%y?????!%y?????B      ??!       J	??a?'????a?'??!??a?'??R      ??!       Z	??a?'????a?'??!??a?'??b      ??!       JGPUY2]Gڳ??b q??'@y?h???V@