"""Minimal Kubernetes operator reconciliation loop for AFETrainingJob resources."""

from __future__ import annotations

import os

from kubernetes import client, config, watch


def reconcile(resource: dict, batch: client.BatchV1Api) -> None:
    spec = resource.get("spec", {})
    metadata = resource["metadata"]
    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=f"afe-{metadata['name']}",
            namespace=metadata.get("namespace", "default"),
        ),
        spec=client.V1JobSpec(
            backoff_limit=2,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[client.V1Container(
                        name="trainer",
                        image=spec.get("image", "ivx:latest"),
                        command=["python", "-m", "engines.training.worker"],
                        env=[client.V1EnvVar(name="AFE_EXPERIMENT_ID", value=spec["experimentId"])],
                    )],
                )
            ),
        ),
    )
    batch.create_namespaced_job(job.metadata.namespace, job)


def main() -> None:
    config.load_incluster_config() if os.getenv("KUBERNETES_SERVICE_HOST") else config.load_kube_config()
    custom = client.CustomObjectsApi()
    batch = client.BatchV1Api()
    for event in watch.Watch().stream(
        custom.list_cluster_custom_object, group="ivx.ai", version="v1alpha1", plural="afetrainingjobs"
    ):
        if event["type"] in {"ADDED", "MODIFIED"}:
            reconcile(event["object"], batch)


if __name__ == "__main__":
    main()
