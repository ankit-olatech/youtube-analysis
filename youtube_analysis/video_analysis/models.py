from django.db import models

class Video(models.Model):
    title = models.CharField(max_length=255, default="NA")
    description = models.TextField(blank=True, null=True)
    upload_date = models.DateTimeField(blank=True, null=True)
    url = models.URLField(max_length=500, default="NA")
    file = models.FileField(upload_to='videos/', blank=True, null=True)  # For manual upload
    views = models.IntegerField(default=0)
    likes = models.IntegerField(default=0)
    dislikes = models.IntegerField(default=0)
    comments_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title