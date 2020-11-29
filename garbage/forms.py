from django import forms
class UploadPictureForm(forms.Form):
	img = forms.ImageField(required=True, label="")

class SerchClassForm(forms.Form):
	word = forms.CharField(max_length=100, label="検索ワード")

class InquiryForm(forms.Form):
	title = forms.CharField(max_length=100, label="タイトル")
	text = forms.CharField(max_length=100, label="内容")
