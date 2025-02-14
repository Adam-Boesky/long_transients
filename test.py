# import google.auth
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from googleapiclient.http import MediaFileUpload


# def upload_basic():
#     """Insert new file.
#     Returns : Id's of the file uploaded

#     Load pre-authorized user credentials from the environment.
#     TODO(developer) - See https://developers.google.com/identity
#     for guides on implementing OAuth2 for the application.
#     """
#     creds, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file'])

#     try:
#         # create drive api client
#         service = build("drive", "v3", credentials=creds)

#         file_metadata = {"name": "/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_both/100_candidate_217p42_20p355.pdf"}
#         media = MediaFileUpload("/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_both/100_candidate_217p42_20p355.pdf")
#         # pylint: disable=maybe-no-member
#         file = (
#             service.files()
#             .create(body=file_metadata, media_body=media, fields="id")
#             .execute()
#         )
#         print(f'File ID: {file.get("id")}')

#     except HttpError as error:
#         print(f"An error occurred: {error}")
#         file = None

#     return file.get("id")


# if __name__ == "__main__":
#   upload_basic()



import google.auth
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_basic():
  """Insert new file.
  Returns : Id's of the file uploaded

  Load pre-authorized user credentials from the environment.
  TODO(developer) - See https://developers.google.com/identity
  for guides on implementing OAuth2 for the application.
  """
  creds = Credentials.from_authorized_user_file("/Users/adamboesky/token.json", SCOPES)

  try:
    # create drive api client
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": "/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_both/100_candidate_217p42_20p355.pdf"}
    media = MediaFileUpload("/Users/adamboesky/Research/long_transients/Data/filter_results/candidates/in_both/100_candidate_217p42_20p355.pdf")
    # pylint: disable=maybe-no-member
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    print(f'File ID: {file.get("id")}')

  except HttpError as error:
    print(f"An error occurred: {error}")
    file = None

  return file.get("id")


if __name__ == "__main__":
  upload_basic()