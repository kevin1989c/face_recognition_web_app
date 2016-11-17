from __future__ import unicode_literals


from django.db import models
class IMG(models.Model):
    img = models.ImageField(upload_to='temp')

# Create your models here.
class Celebr(models.Model):

   
    name=models.CharField(max_length=20)
    content=models.TextField(blank=False)
    mem_id=models.IntegerField(blank=False)
    gender=models.TextField(blank=False)
    img = models.ImageField(upload_to='upload',blank=False)
    created_at=models.DateTimeField(auto_now_add=True)
    def __unicode__(self):
        return self.name

class Celebr_new(models.Model):

   
    name=models.CharField(max_length=20)
    img = models.ImageField(upload_to='new',blank=False)
    created_at=models.DateTimeField(auto_now_add=True)
    def __unicode__(self):
        return self.name



