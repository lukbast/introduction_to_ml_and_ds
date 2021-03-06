?	??ǚ-O@??ǚ-O@!??ǚ-O@      ??!       "h
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
	ǜg?K?	@ǜg?K?	@!ǜg?K?	@      ??!       "	A??! ?L@A??! ?L@!A??! ?L@*      ??!       2      ??!       :	;S輆??;S輆??!;S輆??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?_?e?@y???5W@?"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/expand/Conv2DConv2D??[?n??!??[?n??0"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv/project/Conv2DConv2D?m??Ϋ?!"1????0"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/depthwise/depthwiseDepthwiseConv2dNative??Ƕ!O??!?z?-???"-
IteratorGetNext/_4_Recv????۠?!??n<(???"?
~sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/Conv/Conv2DConv2D??s??E??!???????0"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/depthwise/depthwiseDepthwiseConv2dNative?2"??Ӟ?!%7Є???"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/expand/BatchNorm/FusedBatchNormV3_FusedBatchNormEx2???q??!??$J??"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv/depthwise/depthwiseDepthwiseConv2dNative?-?]{???!??u?;??"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_3/depthwise/depthwiseDepthwiseConv2dNative??W????!?{kĥ???"?
?sequential/keras_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_12/depthwise/depthwiseDepthwiseConv2dNative?1m???!??}???Q      Y@Y???h?@a?k??X@q?U8?OaQ@ysg`?/???"?

both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?69.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Kepler)(: B 