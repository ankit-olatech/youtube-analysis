from django.db import models

# Create your models here.

class Video(models.Model):
    url = models.URLField()
    title = models.CharField(max_length=255)
    description = models.TextField()
    upload_date = models.DateTimeField()
    views = models.IntegerField()
    likes = models.IntegerField()
    dislikes = models.IntegerField()
    comments_count = models.IntegerField()
    category = models.CharField(max_length=100)
    language = models.CharField(max_length=50)