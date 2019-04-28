# Change these as needed and correct command strings below for your
# cloud service provider. This example is for the Google Cloud Platform.
#
# The only variable that should keep its name except the *_COMMAND ones
# below is INSTANCE_NAME_PREFIX which is used to echo which experiment
# belongs to which machine.

IMAGE_FAMILY = 'pytorch-latest-cu92'  # -cu92, maybe -gpu
# Use 'europe-west4-c' for P4, -a for TPUs.
# V100 is available in both.
#
# Or 'europe-west1-b' or -d for highmem but not as
# many GPUs (only P100 and K80).
ZONE = 'europe-west4-c'
# The number at the end is the amount of CPUs.
# 'n1-standard-2' or maybe -4
# or 'n1-highmem-4' or -8.
MACHINE_TYPE = 'n1-standard-4'
GPU_TYPE = 'nvidia-tesla-v100'  # -p4, -v100, -p100 or -k80
# 1, 2, 4, 8.
# Make sure you have enough quota available!
GPU_COUNT = 1
# As we will start more than one machine, this is only a prefix.
# Do not change this name.
INSTANCE_NAME_PREFIX = 'ganerator'
DISK_NAME = 'ganerator-hdd'  # must contain the dataset you want to use

# All the following commands must be a string and will be split at
# spaces (`' '`).
# Also make sure there is a format string indicator (`{{}}`) for the
# instance name suffix after the prefix whenever an instance must be
# called by name.

# How to start your cloud instance.
# To use a startup script, write 'GANERATOR_STARTUP' literally into the
# string to always get the correct path.
# This is so the path can be replaced later due to spaces in paths.
START_COMMAND = (
    'gcloud compute instances create {instance_name_prefix}-{{}} '
        '--zone={zone} '
        '--machine-type={machine_type} '
        '--image-family={image_family} '
        '--image-project=deeplearning-platform-release '
        '--maintenance-policy=TERMINATE '
        '--accelerator="type={gpu_type},count={gpu_count}" '
        '--metadata="install-nvidia-driver=True" '
        '--metadata-from-file startup-script="GANERATOR_STARTUP" '
        '--disk="name={disk_name}" '
        '--preemptible'.format(instance_name_prefix=INSTANCE_NAME_PREFIX,
            zone=ZONE, machine_type=MACHINE_TYPE, image_family=IMAGE_FAMILY,
            gpu_type=GPU_TYPE, gpu_count=GPU_COUNT, disk_name=DISK_NAME)
)

# You must leave two format strings here. One for the command that will
# be executed and one for the instance name suffix.
# Designate them by the format string indicators `{command}` and
# `{suffix}` (using double curly brackets if you want to format the
# string before).
REMOTE_PROCESS_COMMAND = (
    'gcloud compute ssh {instance_name_prefix}-{{suffix}} '
        '--command "{{command}}"'.format(
            instance_name_prefix=INSTANCE_NAME_PREFIX)
)

# This command will be interpolated in the REMOTE_PROCESS_COMMAND to do
# some final initialization in your machine such as navigating to the
# GANerator directory.
# If empty or None, this is skipped.
INIT_COMMAND = (
    ''
)

# How to end or delete your instance.
END_COMMAND = (
    'gcloud compute instances delete {instance_name_prefix}-{{}}'.format(
        instance_name_prefix=INSTANCE_NAME_PREFIX)
)
