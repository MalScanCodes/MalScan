## MalScan: Android Malware Detection Based on Social-Network Centrality Analysis
### Abstract
Malware scanning of an app market is expected to be scalable and effective. However, existing approaches use
syntax-based features that can be evaded by transformation attacks or semantic-based features which are usually extracted by
expensive program analysis. Therefore, to address the scalability challenges of traditional heavyweight static analysis, we propose a
graph-based lightweight approach MalScan for Android malware detection. MalScan considers the function call graph as a complex
social network and employs centrality analysis on sensitive application program interfaces (APIs) to express the semantic
characteristics of the graph. On this basis, machine learning algorithms and ensemble learning algorithms are applied to classify the
extracted features. We evaluate MalScan on datasets of 104,892 benign apps and 108,640 malwares, and the results of experiments
indicate that MalScan outperforms six state-of-the-art detectors and can quickly detect Android malware with an f-value as high as
99%. In addition, there are also significant improvements in the robustness of Android app evolution and robustness to obfuscation.
Finally, we conduct an exhaustive statistical study of over one million applications in the Google-Play app market and successfully
identify 498 zero-day malware, which further validates the feasibility of MalScan on market-wide malware scanning.
