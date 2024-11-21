from django import forms
from multiupload.fields import MultiFileField


class PredictDemandForm(forms.Form):

    files = MultiFileField(min_num=1, max_num=2, max_file_size=1024 * 1024 * 5)

    # file = forms.FileField(
    #     widget=forms.ClearableFileInput(attrs={'multiple': True}),
    #     required=False
    # )  # for creating file input

    # def clean_file(self):
    #     uploaded_files = self.cleaned_data.get('files')

    #     if len(uploaded_files) > 2:
    #         raise forms.ValidationError("You can upload only one or two files. Either 2 files with 24 hours each, or one file with 48 hours of data.")
        
    #     for uploaded_file in uploaded_files:
    #         file_name = uploaded_file.name.lower()
    #         if not file_name.endswith(('.csv', '.xlsx', '.xls')):
    #             raise forms.ValidationError(
    #                 f"Unsupported file type: {file_name}. Please upload a CSV or Excel file."
    #             )
        
    #     return uploaded_files




class UploadActualsForm(forms.Form):
    files = MultiFileField(min_num=1, max_num=2, max_file_size=1024 * 1024 * 5)

    # file = forms.FileField(
    #     widget=forms.ClearableFileInput(attrs={'multiple': True}),
    #     required=False
    # )  # for creating file input

    # def clean_file(self):
    #     uploaded_file = self.cleaned_data.get('file')
    #     if uploaded_file:
    #         # Check if the file type is CSV or Excel (XLSX or XLS)
    #         file_name = uploaded_file.name.lower()
    #         if not file_name.endswith(('.csv', '.xlsx', '.xls')):
    #             raise forms.ValidationError(
    #                 "Unsupported file type. Please upload a CSV or Excel file.")
    #     return uploaded_file


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result