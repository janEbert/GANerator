# If you use more than one instance, pass tuples to assign different
# values. For example, for two instances use zones
# `('europe-west4-a', 'europe-west4-c')`.
NUM_INSTANCES = 8

# These paths are relative to the file `distribute_multi.py` as a tuple
# that will be `os.path.join`ed or single string. Don't forget to wrap
# the tuple or string in another tuple if you only specify one path!

CLOUD_API_PATH = ('gcloud',)
STARTUP_SCRIPT_PATH = ('GANerator_GCP_startup.sh',)


# Change these as needed and correct command strings below for your
# cloud service provider. This example is for the Google Cloud Platform.
#
# If you do not need a variable, simply set it to the empty string.
# To do more editing (in case you need more variables), look into
# `distribute_multi.py`.

IMAGE_FAMILY = 'pytorch-latest-cu92'  # -cu92, maybe -gpu
# Use 'europe-west4-c' for P4, -a for TPUs.
# V100 is available in both.
#
# Or 'europe-west1-b' or -d for highmem but not as
# many GPUs (only P100 and K80).
ZONE = ('europe-west4-a', 'europe-west4-b', 'europe-west4-c', 'europe-west4-c',
        'europe-west1-a', 'europe-west1-b', 'europe-west4-a', 'europe-west4-a')
# The number at the end is the amount of CPUs.
# 'n1-standard-2' or maybe -4
# or 'n1-highmem-4' or -8.
MACHINE_TYPE = 'n1-standard-4'
# -p4, -v100, -p100 or -k80
GPU_TYPE = ('nvidia-tesla-v100',) * 4 + ('nvidia-tesla-p100',) * 4
# 1, 2, 4, 8.
# Make sure you have enough quota available!
GPU_COUNT = 1
# As we will start more than one machine, this is only a prefix.
# Do not change this name.
INSTANCE_NAME_PREFIX = 'ganerator'
DISK_NAME = 'ganerator-ssd'  # must contain the dataset you want to use
# service account you want to use
SERVICE_ACCOUNT = 'ganerator-service-account@ganerator.iam.gserviceaccount.com'

# All the following commands must be a string and will be split at
# spaces (`' '`).
# To properly manage the path for your cloud provider API binary, write
# the string 'GANERATOR_CLOUD_BIN' literally in its place (or just write
# the binary if it is in your PATH).
#
# Leave keyword format string indicators for the arguments (using double
# curly braces if you want to format the string before).
# TODO make interpolation values choosable via another tuple and **kwargs in format function

# How to start your cloud instance.
# To use a startup script, write 'GANERATOR_STARTUP' literally into the
# string to always get the correct path.
# This is so the path can be replaced later due to spaces in paths.
# The following arguments will be interpolated into the same lower-case
# keyword format string indicator:
# INSTANCE_NAME_PREFIX, ZONE, MACHINE_TYPE, IMAGE_FAMILY, GPU_TYPE,
# GPU_COUNT, DISK_NAME, SERVICE_ACCOUNT
#
# Also make sure there is _another_ format string indicator `{suffix}`
# for the instance name suffix after the prefix.
START_COMMAND = (
    'gcloud compute instances create {instance_name_prefix}-{suffix} '
        '--zone={zone} '
        '--machine-type={machine_type} '
        '--image-family={image_family} '
        '--image-project=deeplearning-platform-release '
        '--maintenance-policy=TERMINATE '
        '--accelerator="type={gpu_type},count={gpu_count}" '
        '--metadata="install-nvidia-driver=True" '
        '--metadata-from-file startup-script="GANERATOR_STARTUP" '
        '--disk="name={disk_name}" '
        '--service-account={service_account} '
        '--preemptible'
)

# You must leave one string literal and two format string indicators
# here. The literal is for the command that will be executed and the
# format string indicator for the instance name prefix and suffix.
# Designate the command by the string literal 'GANERATOR_COMMAND' and
# the instance name prefix and suffix by the format string indicators
# `{instance_name_prefix}` and `{suffix}` respectively (using double
# curly braces if you want to format the string before).
REMOTE_PROCESS_COMMAND = (
    'gcloud compute ssh {instance_name_prefix}-{suffix} '
        '--command "GANERATOR_COMMAND"'
)

# This command will be interpolated in the REMOTE_PROCESS_COMMAND to do
# some final initialization in your machine such as navigating to the
# GANerator directory.
# If empty or None, this is skipped.
# You can also interpolate the instance name suffix into the command
# via the format string indicator `{suffix}`.
INIT_COMMAND = (
    ''
)

# This command will be interpolated in the REMOTE_PROCESS_COMMAND to do
# some final work in your machine such as saving your experimental
# results.
# If empty or None, this is skipped.
# You can also interpolate the instance name suffix into the command
# via the format string indicator `{suffix}`.
FINISH_COMMAND = (
    'gsutil cp -r ../GANerator_experiments gs://ganerator-data/instance-{suffix}/'
)

# How to end or delete your instance.
# Leave format string indicators `{instance_name_prefix}` and `{suffix}`
# for the `INSTANCE_NAME_PREFIX` and the corresponding suffix that will
# both be interpolated later.
END_COMMAND = (
    'gcloud compute instances delete {instance_name_prefix}-{suffix}'
)
