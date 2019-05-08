# Change these as needed and correct command strings below for your
# cloud service provider. This example is for the Google Cloud Platform.
#
# The only variable that should keep its name except the *_COMMAND ones
# below is INSTANCE_NAME_PREFIX which is used to echo which experiment
# belongs to which machine.

IMAGE_FAMILY = 'pytorch-latest-cu100'  # -cu100, maybe -gpu
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
GPU_TYPE = 'nvidia-tesla-p4'  # -t4, -p4, -v100, -p100 or -k80
# 1, 2, 4, 8.
# Make sure you have enough quota available!
GPU_COUNT = 1
# As we will start more than one machine, this is only a prefix.
# Do not change this variable's name.
INSTANCE_NAME_PREFIX = 'ganerator'
# Must contain the dataset you want to use.
RO_DISK_NAME = 'ganerator-ssd'
# Service account you want to use.
SERVICE_ACCOUNT = 'ganerator-service-account@ganerator.iam.gserviceaccount.com'

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
        '--create-disk="size=11GB,auto-delete=yes" '
        '--disk="name={ro_disk_name},mode=ro" '
        '--service-account={service_account} '
        '--scopes=storage-full '
        '--preemptible'.format(instance_name_prefix=INSTANCE_NAME_PREFIX,
            zone=ZONE, machine_type=MACHINE_TYPE, image_family=IMAGE_FAMILY,
            gpu_type=GPU_TYPE, gpu_count=GPU_COUNT, ro_disk_name=RO_DISK_NAME,
            service_account=SERVICE_ACCOUNT)
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
# some final initialization in your machine such as cloning and
# navigating to the GANerator directory. This will be executed as a
# direct shell command to allow for more freedom, so escape the correct
# symbols depending on `REMOTE_PROCESS_COMMAND`.
# If empty or None, this is skipped.
# You can also interpolate the instance name suffix into the command
# via the format string indicator `{suffix}`.
INIT_COMMAND = (
    'cd /mnt/disks/rwdisk && '
    'mkdir GANerator_experiments && '
    'git clone -q https://github.com/janEbert/GANerator.git && '
    'cd GANerator && '
    'python3 src/ipynb_to_py.py && '
    'echo \\"cd \\$PWD\\" > ~/.bashrc && '
    'conda init > /dev/null'
)

# This command will be interpolated in the REMOTE_PROCESS_COMMAND to do
# some final work in your machine such as saving your experimental
# results. This will be executed as a direct shell command to allow for
# more freedom, so escape the correct symbols depending on
# `REMOTE_PROCESS_COMMAND`.
# If empty or None, this is skipped.
# You can also interpolate the instance name suffix into the command
# via the format string indicator `{suffix}`.
FINISH_COMMAND = (
    "echo 'Compressing results...' && "
    'export ANAME=\\$(date +%s) && '
    'tar -czf exp-\\$ANAME.tar.gz --remove-files -C .. GANerator_experiments && '
    'gsutil cp exp-\\$ANAME.tar.gz gs://ganerator/ganerator-{suffix}/'
)

# How to end or delete your instance.
END_COMMAND = (
    'gcloud compute instances delete {instance_name_prefix}-{{}} -q'.format(
        instance_name_prefix=INSTANCE_NAME_PREFIX)
)

