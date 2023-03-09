# Generated by Django 4.1.5 on 2023-01-30 16:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('marketBotApp', '0015_remove_adl_date_remove_adx_date_remove_aroon_date_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='finnish_stock_daily',
            name='ask',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='finnish_stock_daily',
            name='average',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='finnish_stock_daily',
            name='bid',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='finnish_stock_daily',
            name='trades',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='finnish_stock_daily',
            name='turnover',
            field=models.FloatField(null=True),
        ),
    ]