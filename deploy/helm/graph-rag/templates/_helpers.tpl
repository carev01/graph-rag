{{- define "graph-rag.labels" -}}
app.kubernetes.io/part-of: graph-rag
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "graph-rag.image" -}}
{{- $tag := required "image.tag is required (sha-<7-char commit>)" .Values.image.tag -}}
{{- if not (regexMatch "^sha-[0-9a-f]{7}$" $tag) -}}
{{- fail (printf "image.tag %q must match the expected format sha-<7-char lowercase-hex commit>, e.g. sha-abc1234" $tag) -}}
{{- end -}}
{{ .Values.image.repository }}:{{ $tag }}
{{- end }}

{{- define "graph-rag.envFrom" -}}
envFrom:
  - secretRef:
      name: {{ .Values.secretName }}
  - configMapRef:
      name: graph-rag-config
{{- end }}

{{- define "graph-rag.podSecurity" -}}
automountServiceAccountToken: false
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  seccompProfile: {type: RuntimeDefault}
{{- end }}

{{- define "graph-rag.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
{{- end }}
