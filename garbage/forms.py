from django import forms
from django.core.mail import EmailMessage


class UploadPictureForm(forms.Form):
	img = forms.ImageField(required=True, label="")


class SerchClassForm(forms.Form):
	word = forms.CharField(max_length=100, label="検索ワード")


class OpinionForm(forms.Form):
	title = forms.CharField(max_length=100, label="件名")
	text = forms.CharField(widget=forms.Textarea, label="内容")

	def send_email(self):
		subject = f"opinion(garbage)"
		message = f"タイトル：{self.title}\n本文：{self.text}"

		to_list = ["cnoibfere@gmail.com"]
		mail = EmailMessage(subject=subject, body=message, to=to_list)
		mail.send()
