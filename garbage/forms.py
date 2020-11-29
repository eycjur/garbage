from django import forms
class UploadPictureForm(forms.Form):
	img = forms.ImageField(required=True, label="")

class SerchClassForm(forms.Form):
	word = forms.CharField(max_length=100, label="検索ワード")