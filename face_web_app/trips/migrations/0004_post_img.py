# -*- coding: utf-8 -*-
# Generated by Django 1.9.8 on 2016-08-10 01:17
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trips', '0003_auto_20160808_2222'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='img',
            field=models.ImageField(blank=True, upload_to='upload'),
        ),
    ]
