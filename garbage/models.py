from django.db import models

# Create your models here.
class Classification(models.Model):
	key = models.TextField(max_length=1000)
	classification = models.TextField(max_length=1000)
