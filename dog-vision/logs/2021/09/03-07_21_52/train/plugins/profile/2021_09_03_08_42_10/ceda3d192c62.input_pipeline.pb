	?E(???@?E(???@!?E(???@	; A?-X@; A?-X@!; A?-X@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?E(???@??x?5	@1D?M?R@ID? 5??YP?I?e[?@r0*	?G?d?=A2R
Iterator::Root::MapAndBatchg|_\?_?@!???!?X@)g|_\?_?@1???!?X@:Preprocessing2[
$Iterator::Root::MapAndBatch::Shuffle ׄ?Ơ??!?(>??)ׄ?Ơ??1?(>??:Preprocessing2E
Iterator::Root???_?@!??&G?X@)e??J?ͦ?1ӝE?b?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 96.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9: A?-X@I????'??Q$?y?̇@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??x?5	@??x?5	@!??x?5	@      ??!       "	D?M?R@D?M?R@!D?M?R@*      ??!       2      ??!       :	D? 5??D? 5??!D? 5??B      ??!       J	P?I?e[?@P?I?e[?@!P?I?e[?@R      ??!       Z	P?I?e[?@P?I?e[?@!P?I?e[?@b      ??!       JGPUY: A?-X@b q????'??y$?y?̇@