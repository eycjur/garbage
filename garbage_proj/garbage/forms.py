from django import forms
class UploadPictureForm(forms.Form):
	img = forms.ImageField(required=True, label="")