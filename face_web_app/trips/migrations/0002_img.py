# -*- coding: utf-8 -*-
# Generated by Django 1.9.8 on 2016-08-08 21:18
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trips', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='IMG',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('imag', models.ImageField(upload_to='upload')),
            ],
        ),
    ]
