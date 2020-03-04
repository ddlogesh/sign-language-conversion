from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
    version='2019-08-06',
    iam_apikey='Sq-WXsXwmOv6Lq8AMai2qj7-2Sw_iVtsJTA7CZ_YbvwM'
)


with open('test2.png', 'rb') as images_file:
    classes = visual_recognition.classify(images_file,threshold='0.6',classifier_ids='DefaultCustomModel_1391466737').get_result()

print(classes)
print(type(classes))
var=classes["images"][0]["classifiers"][0]["classes"][0]["class"]
print("predicted class is ",var)

