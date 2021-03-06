?	@?ի?N@@?ի?N@!@?ի?N@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'@?ի?N@t????N@13ı.n?L@I????????r0*	???(\?]@2R
Iterator::Root::MapAndBatch0?Qd????!??M?G@)0?Qd????1??M?G@:Preprocessing2E
Iterator::Root????Ļ?!???o?V@)Pr?Md???1??G?1F@:Preprocessing2[
$Iterator::Root::MapAndBatch::Shuffleb???I??!??ဂ? @)b???I??1??ဂ? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI ??\?o@QpU4?YW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	t????N@t????N@!t????N@      ??!       "	3ı.n?L@3ı.n?L@!3ı.n?L@*      ??!       2      ??!       :	????????????????!????????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ??\?o@ypU4?YW@?"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/expand/Conv2DConv2D?!Fe\??!?!Fe\??0"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv/project/Conv2DConv2D25??˫?!n+?????0"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/depthwise/depthwiseDepthwiseConv2dNativeMJ???R??!J?-\????"-
IteratorGetNext/_4_Recv??A?????!?V?????"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/Conv/Conv2DConv2D?z???R??!??=?B???0"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/depthwise/depthwiseDepthwiseConv2dNativeI?z ?Ğ?!?NM?????"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_1/expand/BatchNorm/FusedBatchNormV3_FusedBatchNormEx0*??Y??!?!??H??"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv/depthwise/depthwiseDepthwiseConv2dNatived?H^
h??!
SB??"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_3/depthwise/depthwiseDepthwiseConv2dNative??!?????!?n??J???"?
?sequential_1/keras_layer_1/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_12/depthwise/depthwiseDepthwiseConv2dNative|*??4??!6?B????Q      Y@Y???h?@a?k??X@qR?W?Q@y?i?0-??"?

both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?71.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Kepler)(: B 