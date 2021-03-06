?	?C?|0N@?C?|0N@!?C?|0N@	??ve?????ve???!??ve???"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?C?|0N@'?y?S@1~?$A$L@I]?6?ɮ??Y??z?"0??r0*	??MbPY@2E
Iterator::RootsI?v??!?Y???8W@),??????1E????L@:Preprocessing2R
Iterator::Root::MapAndBatch???3??!?????A@)???3??1?????A@:Preprocessing2[
$Iterator::Root::MapAndBatch::Shuffle?g??}?!?a
R?u@)?g??}?1?a
R?u@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??ve???I??ϣ@QGa?;NW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'?y?S@'?y?S@!'?y?S@      ??!       "	~?$A$L@~?$A$L@!~?$A$L@*      ??!       2      ??!       :	]?6?ɮ??]?6?ɮ??!]?6?ɮ??B      ??!       J	??z?"0????z?"0??!??z?"0??R      ??!       Z	??z?"0????z?"0??!??z?"0??b      ??!       JGPUY??ve???b q??ϣ@yGa?;NW@?"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/expand/Conv2DConv2D5A0?????!5A0?????0"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv/project/Conv2DConv2D!?G]Y??!??!?~??0"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/depthwise/depthwiseDepthwiseConv2dNative???;???!J9??8??"-
IteratorGetNext/_3_Send??B&;???!?󑲠???"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/expand/BatchNorm/FusedBatchNormV3_FusedBatchNormEx???$???!:??3ŗ??"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/Conv/Conv2DConv2D????*??!?h=?"}??0"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/depthwise/depthwiseDepthwiseConv2dNative??[???!?p4x?'??"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv/depthwise/depthwiseDepthwiseConv2dNative???	???!?=?S???"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_3/depthwise/depthwiseDepthwiseConv2dNative?G???&??!E"B?????"?
?sequential_2/keras_layer_2/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_11/expand/Conv2DConv2D??
?k??!?c?a{???0Q      Y@Y???h?@a?k??X@q$v????I@y??F6????"?

both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?52.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Kepler)(: B 